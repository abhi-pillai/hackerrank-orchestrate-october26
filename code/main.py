#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ghost-Data Deletion Audit Agent
===============================

Architecture
------------
  THE BRAIN    Gemini (gemini-3.1-flash-lite): reads the request, decides what to investigate,
               emits function calls, and submits the final audit.
  THE HANDS    Python tools the Brain can call: identity resolution, footprint search across all
               evidence (SQL / JSON / logs / txt / PDF / images), activity timelines, read-only SQL,
               visual-evidence Q&A, long-term-memory recall, and `submit_audit`.
  THE NOTEBOOK Short-term: per-case conversation history with compaction.
               Long-term : FAISS vector store (Gemini embeddings) that persists across runs and is
               seeded with the solved sample cases, used as few-shot guidance.
  THE ENGINE   `AgentLoop.run()`: Think -> Act (tool call) -> Observe, repeated until the Brain calls
               `submit_audit` or the iteration limit is hit (then a forced submission, then a
               deterministic fallback).

The LLM decides *what is true* (state of every record at `audit_at`). A deterministic PlanBuilder turns
that judgement into the exact, correctly ordered `steps_json` the grader expects.

Run:   python code/main.py                (full run -> output.csv + log.txt in the repo root)
       python code/main.py --eval-samples (score the agent on the 32 sample cases)
Needs: GEMINI_API_KEY in the environment or in a .env file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # numpy fallback keeps the agent runnable without faiss
    faiss = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
THINKING_LEVEL = "medium"        # minimal | low | medium | high
MAX_ITERATIONS = 14              # safety limit for the Think-Act-Observe loop
MAX_TOOL_CHARS = 9000            # cap on any single tool observation sent back to the Brain
STM_BUDGET_CHARS = 90000         # short-term-memory budget before old observations are compacted
DEFAULT_WORKERS = 6
FEW_SHOT_K = 3

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "dataset").exists() else HERE.parent
DATASET = ROOT / "dataset"
OUTPUT_CSV = ROOT / "output.csv"
LOG_TXT = ROOT / "log.txt"
CACHE_DIR = ROOT / ".agent_cache"
MEMORY_DIR = CACHE_DIR / "memory"
RESULTS_CACHE = CACHE_DIR / "results.jsonl"
VISUAL_CACHE = CACHE_DIR / "visual_evidence.json"

# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}:\d{2}))?")
RID_RE = re.compile(r"^[a-z][a-z_]*_\d+$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
SYS_RE = re.compile(r"_(service|platform|store|storage|cluster|replica)$")
CTX_KEYS = {"captured_at", "exported_at", "data_store", "collection", "source", "origin", "region", "system"}
KV_RE = re.compile(r'(\w+)=("([^"]*)"|\S+)')


def norm_ts(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    m = TS_RE.match(v.strip())
    return f"{m.group(1)}T{m.group(2) or '00:00:00'}Z" if m else None


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def case_sort_key(case_id: str):
    m = re.search(r"(\d+)$", case_id)
    return (case_id[: m.start()], int(m.group(1))) if m else (case_id, 0)


def flatten(obj: Any, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            out[prefix[:-1]] = ", ".join(map(str, obj))
        else:
            for i, x in enumerate(obj):
                out.update(flatten(x, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = obj
    return out


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 40] + f"... [truncated {len(text) - n + 40} chars]"


def nonnull(fields: dict, maxlen: int = 90) -> dict:
    return {k: (str(v)[:maxlen]) for k, v in fields.items() if v not in (None, "", "null", "None")}


# ----------------------------------------------------------------------------------------------
# Usage accounting + run log (log.txt)
# ----------------------------------------------------------------------------------------------
class Usage:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.inp = self.out = self.think = self.embed_calls = 0

    def add(self, resp) -> None:
        um = getattr(resp, "usage_metadata", None)
        with self.lock:
            self.calls += 1
            if um:
                self.inp += um.prompt_token_count or 0
                self.out += um.candidates_token_count or 0
                self.think += getattr(um, "thoughts_token_count", 0) or 0

    def summary(self) -> str:
        n = max(self.calls, 1)
        return (f"model_calls={self.calls} embed_calls={self.embed_calls} input_tokens={self.inp} "
                f"output_tokens={self.out} thinking_tokens={self.think} "
                f"avg_total_tokens_per_call={(self.inp + self.out + self.think) // n}")


class RunLog:
    """Append-only, thread-safe writer for log.txt. Each case is written as one atomic block."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()

    def write(self, text: str) -> None:
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------
# DATA LAYER: identities, activity, evidence index
# ----------------------------------------------------------------------------------------------
class Identities:
    """Current identity snapshot + historical changes -> every identifier a user ever had."""

    def __init__(self):
        self.cur = {r["user_id"]: r for r in read_csv(DATASET / "user_identities.csv")}
        self.audit = read_csv(DATASET / "user_audit_log.csv")
        self.hist: dict[str, list] = defaultdict(list)
        self.email_users: dict[str, set] = defaultdict(set)
        for uid, r in self.cur.items():
            self.email_users[r["email"].strip().lower()].add(uid)
        for a in self.audit:
            self.hist[a["user_id"]].append(a)
            if a["field_name"] == "email":
                for v in (a["old_value"], a["new_value"]):
                    if v:
                        self.email_users[v.strip().lower()].add(a["user_id"])
        self._cache: dict[str, dict] = {}

    def resolve(self, email: str) -> list[str]:
        return sorted(self.email_users.get(email.strip().lower(), set()))

    def profile(self, uid: str) -> Optional[dict]:
        if uid in self._cache:
            return self._cache[uid]
        c = self.cur.get(uid)
        if not c:
            return None
        emails, phones, hashes = {c["email"].lower()}, {c["phone"]}, {c["email_hash"].lower()}
        for a in self.hist.get(uid, []):
            for v in (a["old_value"], a["new_value"]):
                if not v:
                    continue
                if a["field_name"] == "email":
                    emails.add(v.lower())
                elif a["field_name"] == "phone":
                    phones.add(v)
                elif a["field_name"] == "email_hash":
                    hashes.add(v.lower())
        hashes |= {hashlib.sha256(e.encode()).hexdigest() for e in emails}
        p = {
            "user_id": uid, "name": c["name"], "current_email": c["email"], "current_phone": c["phone"],
            "emails": sorted(emails), "phones": sorted(phones), "email_hashes": sorted(hashes),
            "history": [{"audit_id": a["audit_id"], "time": a["event_time"], "field": a["field_name"],
                         "old": a["old_value"], "new": a["new_value"]} for a in self.hist.get(uid, [])],
        }
        p["_match"] = {
            "emails": p["emails"], "hashes": p["email_hashes"],
            "phones": [re.sub(r"\D", "", x) for x in phones if len(re.sub(r"\D", "", x)) >= 8],
            "uid_re": re.compile(rf"(?<![\w]){re.escape(uid)}(?!\d)"),
            "names": [c["name"].lower()] if len(c["name"].split()) >= 2 else [],
        }
        self._cache[uid] = p
        return p


class ActivityIndex:
    def __init__(self):
        self.rows = read_csv(DATASET / "system_activity.csv")
        self.by_id = {r["activity_id"]: r for r in self.rows}
        self.by_resource: dict[str, list] = defaultdict(list)
        self.links_by_user: dict[str, list] = defaultdict(list)
        self.links_by_email: dict[str, list] = defaultdict(list)
        for r in self.rows:
            seen = {r["resource_id"]} | set(re.findall(r"[a-z][a-z_]*_\d+", r["details"]))
            for rid in seen:
                if RID_RE.match(rid):
                    self.by_resource[rid].append(r)
            if r["event_type"] == "LINK_RESOURCE_IDENTITY":
                for rid in set(re.findall(r"[a-z][a-z_]*_\d+", r["details"])) - set(re.findall(r"user_\d+", r["details"])):
                    for uid in re.findall(r"user_\d+", r["details"]):
                        self.links_by_user[uid].append((rid, r))
                    for em in EMAIL_RE.findall(r["details"]):
                        self.links_by_email[em.lower()].append((rid, r))
        for lst in self.by_resource.values():
            lst.sort(key=lambda x: x["event_time"])


class EvidenceIndex:
    """Normalises every evidence file into 'observations' (one record at one point in time)."""

    def __init__(self):
        self.manifest = {r["file_id"]: r for r in read_csv(DATASET / "evidence_files.csv")}
        self.obs: list[dict] = []
        self.by_resource: dict[str, list] = defaultdict(list)
        self.sql: dict[str, sqlite3.Connection] = {}
        self.sql_lock = threading.Lock()
        self.visual: dict[str, dict] = {}
        self.coverage: dict[str, set] = defaultdict(set)   # file_id -> resource-id prefixes the file covers

    def _cover(self, fid: str, system: Any):
        pre = SYSTEM_TO_PREFIX.get(str(system))
        if pre:
            self.coverage[fid].add(pre)

    # ---- add one observation ----
    def _add(self, fid: str, kind: str, ts: Any, rid: Optional[str], fields: dict, collection: str = "", system: str = ""):
        ts = norm_ts(ts) or norm_ts(self.manifest[fid]["captured_at"])
        vals = [v for v in fields.values() if v not in (None, "")]
        ob = {
            "file_id": fid, "kind": kind, "ts": ts, "resource_id": rid or "", "collection": collection,
            "system": system, "fields": fields,
            "_hay": " ".join(str(v).lower() for v in vals),
            "_digits": " ".join(re.sub(r"\D", "", str(v)) for v in vals if isinstance(v, (str, int)) and 0 < len(str(v)) <= 32),
        }
        self.obs.append(ob)
        if rid:
            self.by_resource[rid].append(ob)

    # ---- loaders ----
    def build(self, client=None, log: Optional[RunLog] = None) -> None:
        for fid, row in self.manifest.items():
            path = DATASET / row["file_path"]
            if not path.exists():
                continue
            ext = path.suffix.lower()
            try:
                if ext == ".json":
                    self._walk_json(fid, json.loads(path.read_text(encoding="utf-8")), {}, [])
                elif ext == ".sql":
                    self._load_sql(fid, path)
                elif ext == ".log":
                    self._load_log(fid, path)
                elif ext == ".txt":
                    self._load_txt(fid, path)
            except Exception as e:  # one bad file must never kill the run
                if log:
                    log.write(f"[index] WARN could not parse {row['file_path']}: {e}")
        if client is not None:
            self._load_visuals(client, log)
        for lst in self.by_resource.values():
            lst.sort(key=lambda o: o["ts"] or "")
        # which resource families (id prefixes) each evidence file covers -> lets us prove ABSENCE
        for o in self.obs:
            if o["resource_id"]:
                self.coverage[o["file_id"]].add(prefix_of(o["resource_id"]))

    def _walk_json(self, fid, node, ctx, path):
        if isinstance(node, dict):
            if isinstance(node.get("resource_id"), str):
                f = flatten(node)
                merged = {**{k: v for k, v in ctx.items() if k not in f}, **f}
                ts = f.get("observed_at") or f.get("recorded_at") or ctx.get("captured_at") or ctx.get("exported_at")
                system = ctx.get("data_store") or ctx.get("system") or next((p for p in path if SYS_RE.search(p)), "") or ctx.get("origin", "")
                self._add(fid, "json", ts, node["resource_id"], merged,
                          collection=ctx.get("collection") or (path[-1] if path else ""), system=system)
                return
            new = dict(ctx)
            new.update({k: v for k, v in node.items() if k in CTX_KEYS and not isinstance(v, (dict, list))})
            if node.get("data_store"):
                self._cover(fid, node["data_store"])
            for k, v in node.items():
                if isinstance(v, dict) and SYS_RE.search(k):
                    self._cover(fid, k)
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    self._walk_json(fid, v, new, path + [k])
        elif isinstance(node, list):
            for x in node:
                self._walk_json(fid, x, ctx, path)

    def _load_sql(self, fid, path: Path):
        text = path.read_text(encoding="utf-8", errors="replace")
        con = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            con.executescript(text)
        except sqlite3.Error:  # fall back to statement-by-statement
            con.close()
            con = sqlite3.connect(":memory:", check_same_thread=False)
            buf = ""
            for line in text.splitlines():
                buf += line + "\n"
                if sqlite3.complete_statement(buf):
                    try:
                        con.execute(buf)
                    except sqlite3.Error:
                        pass
                    buf = ""
        con.row_factory = sqlite3.Row
        for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]
            if "resource_id" not in cols:
                continue
            for row in con.execute(f'SELECT * FROM "{table}"'):
                d = dict(row)
                extra: dict = {}
                for k, v in list(d.items()):  # expand JSON blobs such as record_json
                    if isinstance(v, str) and v.startswith("{"):
                        try:
                            extra.update(flatten(json.loads(v)))
                            d[k] = "(expanded)"
                        except ValueError:
                            pass
                fields = {**extra, **d}
                self._add(fid, "sql", None, d["resource_id"], fields, collection=table,
                          system=d.get("origin_system") or d.get("data_store") or "")
        try:
            for (osys,) in con.execute("SELECT origin_system FROM snapshot_scope").fetchall():
                self._cover(fid, osys)
        except sqlite3.Error:
            pass
        con.execute("PRAGMA query_only=ON")
        self.sql[fid] = con

    def _load_log(self, fid, path: Path):
        traces: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s+(.*)$", line)
            if not m:
                continue
            ts, rest = m.groups()
            kv = {k: (q if full.startswith('"') else full) for k, full, q in KV_RE.findall(rest)}
            kv = {k: v for k, v in kv.items() if v != "null"}
            tid = kv.get("trace_id")
            if tid:
                t = traces.setdefault(tid, {"ts": ts, "fields": {}, "events": []})
                t["ts"] = ts
                t["events"].append(kv.get("event", ""))
                t["fields"].update(kv)
            elif "resource_id" in kv or "@" in rest:
                self._add(fid, "log", ts, kv.get("resource_id"), {**kv, "raw": line[:300]},
                          system=kv.get("service", ""))
        for t in traces.values():
            f = t["fields"]
            f["events"] = ",".join(e for e in t["events"] if e)
            if f.get("resource_id") or "@" in " ".join(map(str, f.values())):
                self._add(fid, "log", t["ts"], f.get("resource_id"), f, system=f.get("service", ""))

    def _load_txt(self, fid, path: Path):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [p.strip() for p in line.lstrip("- ").split("|")]
            if len(parts) >= 5 and RID_RE.match(parts[2]):
                self._add(fid, "txt", parts[0], parts[2],
                          {"system_id": parts[1], "category": parts[3], "result": parts[4],
                           "identifier": " | ".join(parts[5:])}, system=parts[1])
            elif "@" in line:
                self._add(fid, "txt", None, None, {"raw": line[:300]})

    # ---- multimodal evidence (PDF / PNG) transcribed once by Gemini and cached ----
    def _load_visuals(self, client, log):
        cache = json.loads(VISUAL_CACHE.read_text()) if VISUAL_CACHE.exists() else {}
        schema = {
            "type": "object",
            "properties": {
                "page_text": {"type": "string", "description": "Verbatim transcription of all visible text."},
                "records": {"type": "array", "items": {"type": "object", "properties": {
                    "resource_id": {"type": "string"}, "system": {"type": "string"},
                    "collection": {"type": "string"}, "observed_at": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "value": {"type": "string"}}, "required": ["name", "value"]}}},
                    "required": ["resource_id"]}}},
            "required": ["page_text", "records"],
        }
        prompt = ("Transcribe this evidence artefact exactly. Return every visible record with its resource_id "
                  "(e.g. account_123, search_45), system, timestamp and every field value (email, phone, name, "
                  "email_hash, status, enabled, restore_allowed, retention_until, ...). Write 'null' for blank/"
                  "masked fields. Do not infer; copy what you see.")
        for fid, row in self.manifest.items():
            path = DATASET / row["file_path"]
            if path.suffix.lower() not in (".png", ".pdf") or not path.exists():
                continue
            if fid not in cache:
                try:
                    mime = "application/pdf" if path.suffix.lower() == ".pdf" else "image/png"
                    resp = client.models.generate_content(
                        model=MODEL,
                        contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), prompt],
                        config={"response_mime_type": "application/json", "response_json_schema": schema},
                    )
                    cache[fid] = json.loads(resp.text)
                    if log:
                        log.write(f"[index] transcribed {row['file_path']} ({len(cache[fid].get('records', []))} records)")
                except Exception as e:
                    cache[fid] = {"page_text": "", "records": [], "error": str(e)[:200]}
                    if log:
                        log.write(f"[index] WARN visual transcription failed for {row['file_path']}: {e}")
            data = cache[fid]
            self.visual[fid] = data
            for rec in data.get("records", []):
                fields = {p["name"]: p["value"] for p in rec.get("fields", []) if p.get("value") not in ("null", "")}
                self._add(fid, "visual", rec.get("observed_at"), rec.get("resource_id"), fields,
                          collection=rec.get("collection", ""), system=rec.get("system", ""))
            if data.get("page_text"):
                self._add(fid, "visual_text", None, None, {"raw": data["page_text"][:4000]})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        VISUAL_CACHE.write_text(json.dumps(cache, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------
# Policy knowledge (static hints; the Brain still has to apply them to evidence)
# ----------------------------------------------------------------------------------------------
PREFIX_INFO = {
    "account": ("account_service", "USER_PROFILE", "DELETE (if gone: OK_ABSENT)"),
    "search": ("search_service", "SEARCH_INDEX_ENTRY", "DELETE"),
    "cache": ("cache_service", "CACHE_ENTRY", "DELETE"),
    "support": ("support_platform", "SUPPORT_MESSAGE or SUPPORT_ATTACHMENT",
                "message with shared customer/agent/order context -> REDACT; attachment belonging only to the subject -> DELETE"),
    "analytics": ("analytics_platform", "ANALYTICS_PROFILE / PRODUCT_TELEMETRY", "ANONYMIZE if still attributable to a user"),
    "email_delivery": ("email_delivery_service", "EMAIL_DELIVERY_LOG", "REDACT if it contains an email address"),
    "export": ("export_service", "USER_EXPORT", "DELETE"),
    "object": ("object_storage", "stored export/attachment object", "DELETE"),
    "marketing": ("marketing_platform", "MARKETING_PROFILE", "DELETE"),
    "audit_record": ("security_service", "SECURITY_AUDIT_LOG", "RETAIN while retention_until >= audit_at"),
    "legal_hold": ("legal_service", "LEGAL_HOLD_RECORD", "RETAIN while hold is active"),
    "backup": ("backup_service", "BACKUP_ARCHIVE", "RETAIN archive but restore access must be BLOCKED"),
    "data_flow": ("integration_service", "data flow / connector", "DISABLE if enabled and it can copy/recreate user data"),
    "tombstone": ("identity_service", "ACCOUNT_TOMBSTONE", "REDACT: keep only stable user_id; name/email/phone/email_hash must be gone"),
}
SYSTEM_TO_PREFIX = {v[0]: k for k, v in PREFIX_INFO.items()}
PII_LEAVES = {"email", "phone", "full_name", "display_name", "name", "email_hash", "requester_name", "requester_email",
              "requester_phone", "owner_email", "owner_phone", "customer_email", "identifier", "to", "recipient"}
ABSENT_WORDS = {"not_found", "absent", "deleted_confirmed", "miss", "no_match"}


def prefix_of(rid: str) -> str:
    return re.sub(r"_\d+$", "", rid)


def hint_for(ob: dict) -> dict:
    """Deterministic cues that help (never replace) the Brain's judgement."""
    f = ob["fields"]
    leaves = {k.split(".")[-1].lower(): v for k, v in f.items() if v not in (None, "", "null", "None")}
    pii = sorted(k for k in leaves if k in PII_LEAVES)
    absent = (str(f.get("status_code")) == "404" or str(f.get("result", "")).lower() in ABSENT_WORDS
              or str(f.get("rows")) == "0" or "NOT_FOUND" in str(f.get("result", "")).upper())
    h: dict = {"pii_fields_present": pii, "looks_absent": bool(absent)}
    for key in ("enabled", "restore_allowed", "download_available", "retention_until", "retained_until", "account_status"):
        for k, v in f.items():
            if k.split(".")[-1] == key:
                h[key] = v
    return h


def rid_info(rid: str) -> dict:
    info = PREFIX_INFO.get(prefix_of(rid))
    return {"system_id": info[0], "category": info[1], "policy": info[2]} if info else {}


# ----------------------------------------------------------------------------------------------
# THE NOTEBOOK: short-term and long-term memory
# ----------------------------------------------------------------------------------------------
class ShortTermMemory:
    """Live history of the current case. Old bulky observations are compacted to stay inside budget."""

    PINNED = {"resolve_identity", "find_user_footprint"}

    def __init__(self, budget_chars: int = STM_BUDGET_CHARS):
        self.contents: list[types.Content] = []
        self.budget = budget_chars

    def add_user_text(self, text: str):
        self.contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

    def add_model(self, content: types.Content):
        self.contents.append(content)  # kept verbatim: preserves Gemini thought signatures

    def add_tool_results(self, parts: list[types.Part]):
        self.contents.append(types.Content(role="user", parts=parts))

    def _size(self) -> int:
        n = 0
        for c in self.contents:
            for p in c.parts or []:
                if p.text:
                    n += len(p.text)
                if p.function_response:
                    n += len(json.dumps(p.function_response.response, default=str))
                if p.function_call:
                    n += len(json.dumps(dict(p.function_call.args or {}), default=str))
        return n

    def compact(self):
        if self._size() <= self.budget:
            return
        for c in self.contents[:-2]:
            if c.role != "user":
                continue
            for i, p in enumerate(c.parts or []):
                fr = p.function_response
                if fr and fr.name not in self.PINNED:
                    s = json.dumps(fr.response, default=str)
                    if len(s) > 400:
                        c.parts[i] = types.Part.from_function_response(
                            name=fr.name, response={"result": "[older observation omitted to save context; re-run the tool if needed]"})
            if self._size() <= self.budget:
                return


class LongTermMemory:
    """FAISS vector store over Gemini embeddings; persists in .agent_cache/memory across runs."""

    def __init__(self, client, usage: Usage, log: RunLog):
        self.client, self.usage, self.log = client, usage, log
        self.meta: list[dict] = []
        self.vecs: Optional[np.ndarray] = None
        self.index = None
        self.enabled = True
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path, self.vec_path = MEMORY_DIR / "ltm_meta.json", MEMORY_DIR / "ltm_vecs.npy"
        if self.meta_path.exists() and self.vec_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.vecs = np.load(self.vec_path)
            self._rebuild()

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 50):
            r = self.client.models.embed_content(model=EMBED_MODEL, contents=texts[i:i + 50],
                                                 config=types.EmbedContentConfig(task_type=task))
            self.usage.embed_calls += 1
            out += [e.values for e in r.embeddings]
        v = np.array(out, dtype="float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _rebuild(self):
        if self.vecs is None or not len(self.vecs):
            self.index = None
            return
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.vecs.shape[1])
            self.index.add(self.vecs)

    def add(self, items: list[dict]):
        """items: {source, case_id, text, payload}. Replaces earlier entries with the same (source, case_id)."""
        if not self.enabled or not items:
            return
        try:
            keys = {(i["source"], i["case_id"]) for i in items}
            keep = [j for j, m in enumerate(self.meta) if (m["source"], m["case_id"]) not in keys]
            old_v = self.vecs[keep] if (self.vecs is not None and keep) else None
            new_v = self._embed([i["text"] for i in items], "RETRIEVAL_DOCUMENT")
            self.meta = [self.meta[j] for j in keep] + items
            self.vecs = new_v if old_v is None else np.vstack([old_v, new_v])
            self._rebuild()
            self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
            np.save(self.vec_path, self.vecs)
        except Exception as e:
            self.enabled = False
            self.log.write(f"[memory] WARN long-term memory disabled: {e}")

    def search(self, query: str, k: int = 3, sources: Optional[set] = None, exclude_case: Optional[str] = None) -> list[dict]:
        if not self.enabled or self.vecs is None or not len(self.vecs):
            return []
        try:
            q = self._embed([query], "RETRIEVAL_QUERY")
            n = min(len(self.meta), k * 6 + 6)
            if self.index is not None:
                _, idx = self.index.search(q, n)
                order = [int(i) for i in idx[0] if i >= 0]
            else:
                order = list(np.argsort(-(self.vecs @ q[0]))[:n])
            res = []
            for j in order:
                m = self.meta[j]
                if (sources and m["source"] not in sources) or m["case_id"] == exclude_case:
                    continue
                res.append(m)
                if len(res) >= k:
                    break
            return res
        except Exception as e:
            self.log.write(f"[memory] WARN recall failed: {e}")
            return []


def summarize_steps(steps: list[dict], limit: int = 900) -> str:
    parts = [f"{s['operation']} {s['system_id']}/{s['resource_id']}"
             for s in steps if s["phase"] in ("FINDING",) or s["operation"] in ("VERIFY_RECORD_PRESENT", "VERIFY_RECORD_ABSENT")]
    return clip("; ".join(parts) or "(no steps)", limit)


def seed_long_term_memory(ltm: LongTermMemory):
    if any(m["source"] == "gold" for m in ltm.meta):
        return
    gold = {r["case_id"]: r for r in read_csv(DATASET / "sample_output.csv")}
    items = []
    for r in read_csv(DATASET / "sample_deletion_requests.csv"):
        g = gold.get(r["case_id"])
        if not g:
            continue
        steps = json.loads(g["steps_json"])
        items.append({"source": "gold", "case_id": r["case_id"],
                      "text": f"[{r['reported_status']}] {r['request_text']}",
                      "payload": {"status": g["deletion_status"], "reported_status": r["reported_status"],
                                  "key_steps": summarize_steps(steps), "n_steps": len(steps)}})
    ltm.add(items)


# ----------------------------------------------------------------------------------------------
# DETERMINISTIC PLAN BUILDER: Brain's per-record judgement -> exact steps_json
# ----------------------------------------------------------------------------------------------
FINDING_OPS = ["REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII", "REPORT_ACTIVE_DATA_FLOW", "REPORT_RESTORE_RISK"]
REMED_OPS = ["DISABLE_DATA_FLOW", "BLOCK_RESTORE", "DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD"]
VERIFY_OPS = ["VERIFY_RECORD_ABSENT", "VERIFY_RECORD_PRESENT", "VERIFY_RECORD_REDACTED", "VERIFY_RECORD_ANONYMIZED",
              "VERIFY_DATA_FLOW_DISABLED", "VERIFY_RESTORE_BLOCKED"]
# state_at_audit -> (finding_op+state, remediation_op+state, verification_op+state)
STATE_TABLE = {
    "REMAINING_RECORD":      (("REPORT_REMAINING_RECORD", "PRESENT"), ("DELETE_RECORD", "ABSENT"), ("VERIFY_RECORD_ABSENT", "ABSENT")),
    "EXPOSED_PII_REDACT":    (("REPORT_EXPOSED_PII", "PRESENT"), ("REDACT_RECORD", "REDACTED"), ("VERIFY_RECORD_REDACTED", "REDACTED")),
    "EXPOSED_PII_ANONYMIZE": (("REPORT_EXPOSED_PII", "PRESENT"), ("ANONYMIZE_RECORD", "ANONYMIZED"), ("VERIFY_RECORD_ANONYMIZED", "ANONYMIZED")),
    "ACTIVE_DATA_FLOW":      (("REPORT_ACTIVE_DATA_FLOW", "ACTIVE"), ("DISABLE_DATA_FLOW", "DISABLED"), ("VERIFY_DATA_FLOW_DISABLED", "DISABLED")),
    "RESTORE_RISK":          (("REPORT_RESTORE_RISK", "ACTIVE"), ("BLOCK_RESTORE", "BLOCKED"), ("VERIFY_RESTORE_BLOCKED", "BLOCKED")),
    "OK_ABSENT":             (None, None, ("VERIFY_RECORD_ABSENT", "ABSENT")),
    "OK_RETAINED":           (None, None, ("VERIFY_RECORD_PRESENT", "PRESENT")),
    "OK_REDACTED":           (None, None, ("VERIFY_RECORD_REDACTED", "REDACTED")),
    "OK_ANONYMIZED":         (None, None, ("VERIFY_RECORD_ANONYMIZED", "ANONYMIZED")),
    "OK_FLOW_DISABLED":      (None, None, ("VERIFY_DATA_FLOW_DISABLED", "DISABLED")),
    "OK_RESTORE_BLOCKED":    (None, None, ("VERIFY_RESTORE_BLOCKED", "BLOCKED")),
}
PROBLEM_STATES = {k for k, v in STATE_TABLE.items() if v[0]}


def build_plan(deletion_requested: bool, records: list[dict]) -> tuple[str, list[dict]]:
    if not deletion_requested:
        return "NOT_APPLICABLE", []
    status = "INCOMPLETE" if any(r["state_at_audit"] in PROBLEM_STATES for r in records) else "COMPLETE"
    seen, findings, remeds, verifs = set(), [], [], []
    for r in records:
        key = (r["system_id"], r["resource_id"], r["state_at_audit"])
        if key in seen:
            continue
        seen.add(key)
        f, m, v = STATE_TABLE[r["state_at_audit"]]
        ev = r.get("evidence_id", "")
        if f:
            findings.append((f, r, ev))
        if m:
            remeds.append((m, r, ""))
        verifs.append((v, r, ev if status == "COMPLETE" else ""))

    def order(group, ops):
        return sorted(group, key=lambda x: (ops.index(x[0][0]), x[1]["system_id"], x[1]["resource_id"]))

    steps = []
    for phase, group, ops in (("FINDING", findings, FINDING_OPS), ("REMEDIATION", remeds, REMED_OPS),
                              ("VERIFICATION", verifs, VERIFY_OPS)):
        if status == "COMPLETE" and phase != "VERIFICATION":
            continue
        for (op, state), r, ev in order(group, ops):
            steps.append({"step_no": len(steps) + 1, "phase": phase, "system_id": r["system_id"],
                          "resource_id": r["resource_id"], "operation": op, "state": state, "evidence_id": ev})
    return status, steps


# ----------------------------------------------------------------------------------------------
# THE HANDS: tools the Brain can call (one ToolBox per case => audit_at is enforced everywhere)
# ----------------------------------------------------------------------------------------------
class Shared:
    """Read-only objects shared by all workers."""

    def __init__(self, client, usage: Usage, log: RunLog):
        self.client, self.usage, self.log = client, usage, log
        self.ids = Identities()
        self.activity = ActivityIndex()
        self.ev = EvidenceIndex()
        self.rules = read_csv(DATASET / "deletion_rules.csv")
        self.audit_ids = {a["audit_id"] for a in self.ids.audit}
        self.ltm: Optional[LongTermMemory] = None

    def valid_evidence_ids(self) -> set:
        return set(self.ev.manifest) | set(self.activity.by_id) | self.audit_ids


class ToolBox:
    def __init__(self, shared: Shared, case: dict):
        self.s, self.case = shared, case
        self.audit_at = norm_ts(case["audit_at"]) or "9999-12-31T23:59:59Z"
        self.submission: Optional[dict] = None
        self.known_resources: set[str] = set()

    # -- dispatcher --
    def run(self, name: str, args: dict) -> Any:
        fn = getattr(self, f"t_{name}", None)
        if not fn:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # -- helpers --
    def _obs_before(self, rid: str) -> list[dict]:
        return [o for o in self.s.ev.by_resource.get(rid, []) if (o["ts"] or "") <= self.audit_at]

    def _acts_before(self, rid: str) -> list[dict]:
        return [a for a in self.s.activity.by_resource.get(rid, []) if a["event_time"] <= self.audit_at]

    @staticmethod
    def _compact_obs(o: dict) -> dict:
        return {"evidence_id": o["file_id"], "time": o["ts"], "kind": o["kind"], "collection": o["collection"],
                "system": o["system"], "fields": nonnull(o["fields"]), "hint": hint_for(o)}

    @staticmethod
    def _compact_act(a: dict) -> dict:
        return {"activity_id": a["activity_id"], "time": a["event_time"], "type": a["event_type"],
                "route": f"{a['source_system']}->{a['target_system']}", "reported_result": a["reported_result"],
                "details": a["details"][:160]}

    @staticmethod
    def _match(o: dict, m: dict) -> list[str]:
        r = []
        if any(e in o["_hay"] for e in m["emails"]):
            r.append("email")
        if any(h in o["_hay"] for h in m["hashes"]):
            r.append("email_hash")
        if m["phones"] and any(p in o["_digits"] for p in m["phones"]):
            r.append("phone")
        if m["uid_re"].search(o["_hay"]):
            r.append("user_ref")
        if not r and any(n in o["_hay"] for n in m["names"]):
            r.append("name(weak)")
        return r

    # -- tools --
    def t_get_deletion_rules(self) -> dict:
        return {"rules": self.s.rules,
                "resource_prefix_guide": {p: {"system_id": v[0], "category": v[1], "policy": v[2]} for p, v in PREFIX_INFO.items()}}

    def t_resolve_identity(self, email: str) -> dict:
        uids = self.s.ids.resolve(email)
        if not uids:
            return {"matches": [], "note": "no user owns this email (current or historical)"}
        out = []
        for u in uids:
            p = dict(self.s.ids.profile(u))
            p.pop("_match", None)
            out.append(p)
        return {"matches": out, "note": "multiple users share this email historically; use the one whose history fits"
                if len(out) > 1 else "ok"}

    def t_find_user_footprint(self, user_id: str) -> dict:
        p = self.s.ids.profile(user_id)
        if not p:
            return {"error": f"unknown user_id {user_id}"}
        m = p["_match"]
        why: dict[str, set] = defaultdict(set)
        unlinked = []
        for o in self.s.ev.obs:
            if (o["ts"] or "") > self.audit_at:
                continue
            reasons = self._match(o, m)
            if not reasons:
                continue
            if o["resource_id"]:
                why[o["resource_id"]].add(f"{'+'.join(reasons)}@{o['file_id']}")
            elif len(unlinked) < 8:
                unlinked.append({"evidence_id": o["file_id"], "matched": reasons, "text": str(o["fields"].get("raw", ""))[:200]})
        links = list(self.s.activity.links_by_user.get(user_id, []))
        for e in p["emails"]:
            links += self.s.activity.links_by_email.get(e, [])
        for rid, a in links:
            if a["event_time"] <= self.audit_at:
                why[rid].add(f"catalog_link@{a['activity_id']}")
        resources = []
        for rid in sorted(why, key=lambda r: (prefix_of(r), int(re.search(r"(\d+)$", r).group(1)))):
            obs, acts = self._obs_before(rid), self._acts_before(rid)
            resources.append({
                "resource_id": rid, **rid_info(rid), "linked_via": sorted(why[rid])[:6],
                "latest_observations": [self._compact_obs(o) for o in obs[-3:]],
                "recent_activity": [self._compact_act(a) for a in acts[-4:]],
            })
        self.known_resources |= {r["resource_id"] for r in resources}
        return {"user_id": user_id, "audit_at": self.audit_at, "resource_count": len(resources),
                "resources": resources, "unlinked_text_hits": unlinked,
                "timing_candidates": self._timing_candidates({r["resource_id"] for r in resources}),
                "note": "Only evidence at or before audit_at is shown. 'resources' are linked by identifiers/catalog. "
                        "'timing_candidates' are deletion-worker actions logged around the request's reported_at: ownership is UNVERIFIED "
                        "(there are decoys for other users) - accept one only if it fits this user (closest to reported_at, no other user's "
                        "identifiers on it) and confirm with check_absence / get_resource_timeline."}

    def _timing_candidates(self, already: set) -> list[dict]:
        rep = datetime.strptime(norm_ts(self.case["reported_at"]), "%Y-%m-%dT%H:%M:%SZ")
        out = []
        for a in self.s.activity.rows:
            if a["source_system"] != "deletion_worker" or a["resource_id"] in already or a["event_time"] > self.audit_at:
                continue
            delta = (datetime.strptime(a["event_time"], "%Y-%m-%dT%H:%M:%SZ") - rep).total_seconds() / 60
            if -120 <= delta <= 30:
                out.append({"resource_id": a["resource_id"], **rid_info(a["resource_id"]), "minutes_from_reported_at": round(delta),
                            "activity": self._compact_act(a), "evidence_rows_exist": bool(self._obs_before(a["resource_id"]))})
        out.sort(key=lambda x: abs(x["minutes_from_reported_at"]))
        return out[:14]

    def t_check_absence(self, resource_id: str) -> dict:
        pre = prefix_of(resource_id)
        files = [f for f, cov in self.s.ev.coverage.items()
                 if pre in cov and norm_ts(self.s.ev.manifest[f]["captured_at"]) <= self.audit_at]
        files.sort(key=lambda f: self.s.ev.manifest[f]["captured_at"], reverse=True)
        rows = []
        for f in files:
            here = [o for o in self.s.ev.by_resource.get(resource_id, []) if o["file_id"] == f]
            marker = [o for o in here if hint_for(o)["looks_absent"]]
            rows.append({"evidence_id": f, "captured_at": self.s.ev.manifest[f]["captured_at"],
                         "origin": self.s.ev.manifest[f]["origin"],
                         "kind": here[0]["kind"] if here else "snapshot",
                         "resource_present": len(here) > len(marker),
                         "explicit_not_found_marker": bool(marker),
                         "rows": [self._compact_obs(o) for o in here[:3]]})
        # strongest proof first: files showing the resource gone (explicit 404 marker, then silent coverage), latest first;
        # files where the data is still present come after
        rows.sort(key=lambda r: (r["resource_present"], not r["explicit_not_found_marker"], [-ord(ch) for ch in r["captured_at"]]))
        rows = rows[:8]
        return {"resource_id": resource_id, **rid_info(resource_id),
                "covering_evidence_best_first": rows,
                "activity": [self._compact_act(a) for a in self._acts_before(resource_id)[-6:]],
                "note": "A file that covers this resource family but has no row for the resource (resource_present=false) is evidence of ABSENCE "
                        "at its captured_at; prefer the latest such file at or before audit_at. resource_present=true with identifiers means the data still exists."}

    def t_get_resource_timeline(self, resource_id: str) -> dict:
        obs, acts = self._obs_before(resource_id), self._acts_before(resource_id)
        self.known_resources.add(resource_id)
        return {"resource_id": resource_id, **rid_info(resource_id),
                "observations": [self._compact_obs(o) for o in obs[-25:]],
                "activity": [self._compact_act(a) for a in acts[-25:]]}

    def t_search_evidence(self, query: str) -> dict:
        q = query.strip().lower()
        hits = [o for o in self.s.ev.obs if q in o["_hay"] and (o["ts"] or "") <= self.audit_at]
        return {"query": query, "total_hits": len(hits),
                "hits": [{"resource_id": o["resource_id"], **self._compact_obs(o)} for o in hits[:15]]}

    def t_query_sql_snapshot(self, file_id: str, sql: str) -> dict:
        con = self.s.ev.sql.get(file_id)
        if con is None:
            return {"error": f"{file_id} is not a SQL snapshot"}
        if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
            return {"error": "only a single read-only SELECT is allowed"}
        with self.s.ev.sql_lock:
            cur = con.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(30)]
        return {"file_id": file_id, "captured_at": self.s.ev.manifest[file_id]["captured_at"],
                "rows": [nonnull(r, 120) for r in rows]}

    def t_read_visual_evidence(self, file_id: str, question: str = "") -> dict:
        row = self.s.ev.manifest.get(file_id)
        if not row or Path(row["file_path"]).suffix.lower() not in (".png", ".pdf"):
            return {"error": f"{file_id} is not an image/PDF evidence file"}
        if not question:
            v = self.s.ev.visual.get(file_id, {})
            return {"file_id": file_id, "captured_at": row["captured_at"], "transcription": clip(json.dumps(v, ensure_ascii=False), 5000)}
        path = DATASET / row["file_path"]
        mime = "application/pdf" if path.suffix.lower() == ".pdf" else "image/png"
        resp = self.s.client.models.generate_content(
            model=MODEL, contents=[types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), question])
        self.s.usage.add(resp)
        return {"file_id": file_id, "answer": clip(resp.text or "", 2500)}

    def t_recall_similar_cases(self, query: str) -> dict:
        if not self.s.ltm:
            return {"cases": [], "note": "long-term memory unavailable"}
        res = self.s.ltm.search(query, k=3, exclude_case=self.case["case_id"])
        return {"cases": [{"source": m["source"], "case_id": m["case_id"], "request": m["text"][:250], **m["payload"]} for m in res],
                "note": "'gold' entries are verified answers; 'agent' entries are unverified earlier predictions."}

    def t_submit_audit(self, deletion_requested: bool, rationale: str, records: Optional[list] = None) -> dict:
        records = records or []
        errors, clean = [], []
        valid_ids = self.s.valid_evidence_ids()
        for i, r in enumerate(records):
            st = r.get("state_at_audit")
            if st not in STATE_TABLE:
                errors.append(f"records[{i}]: state_at_audit must be one of {sorted(STATE_TABLE)}")
                continue
            if not r.get("system_id") or not r.get("resource_id"):
                errors.append(f"records[{i}]: system_id and resource_id are required")
                continue
            clean.append({"system_id": r["system_id"].strip(), "resource_id": r["resource_id"].strip(),
                          "state_at_audit": st, "evidence_id": (r.get("evidence_id") or "").strip(), "note": r.get("note", "")})
        if deletion_requested and not clean and not errors:
            errors.append("deletion_requested=true but records is empty; list every relevant record incl. already-absent/retained ones")
        if deletion_requested:
            status, _ = build_plan(True, clean)
            for r in clean:
                needs_ev = r["state_at_audit"] in PROBLEM_STATES or status == "COMPLETE"
                if needs_ev and r["evidence_id"] not in valid_ids:
                    errors.append(f"{r['resource_id']}: evidence_id '{r['evidence_id']}' is missing/invalid "
                                  "(use an evidence file_id, activity_id or audit_id that proves this state)")
        if errors:
            return {"accepted": False, "errors": errors[:12]}
        self.submission = {"deletion_requested": bool(deletion_requested), "rationale": rationale, "records": clean}
        return {"accepted": True}


def _decl(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}


S = {"type": "string"}
TOOL_DECLARATIONS = [
    _decl("get_deletion_rules", "Return deletion_rules.csv plus a guide mapping resource-id prefixes to system_id/category/policy.", {}, []),
    _decl("resolve_identity", "Resolve an email (current OR historical) to user_id(s) with every email, phone, email hash and name the user has ever had, plus audit history.",
          {"email": S}, ["email"]),
    _decl("find_user_footprint", "Find every resource in all evidence that exposes or is linked to this user (by any historical email/phone/hash/user_id or catalog link), with latest observations and recent activity, all limited to audit_at.",
          {"user_id": S}, ["user_id"]),
    _decl("get_resource_timeline", "Full chronological evidence observations and system activity for one resource_id up to audit_at. Use it to detect deletes followed by syncs/restores and replica drift.",
          {"resource_id": S}, ["resource_id"]),
    _decl("check_absence", "For one resource_id: list the latest evidence snapshots (<= audit_at) that cover its system and say whether the resource row is present in each. Use it to prove a record is gone (or still present on a replica) and to pick the evidence_id.",
          {"resource_id": S}, ["resource_id"]),
    _decl("search_evidence", "Case-insensitive substring search over all evidence text (email, phone, hash, name, resource id).", {"query": S}, ["query"]),
    _decl("query_sql_snapshot", "Run one read-only SELECT against a SQL snapshot evidence file (SQLite) to compare primary vs replica tables.",
          {"file_id": S, "sql": S}, ["file_id", "sql"]),
    _decl("read_visual_evidence", "Read a PDF/PNG evidence file: returns its cached transcription, or answers a specific `question` by looking at the file again.",
          {"file_id": S, "question": S}, ["file_id"]),
    _decl("recall_similar_cases", "Search long-term memory (FAISS) for similar earlier cases and their verified/unverified outcomes.", {"query": S}, ["query"]),
    _decl("submit_audit", "FINAL ACTION. Submit the audit. List EVERY relevant record with its state at audit_at (including already-absent, retained, redacted, anonymized, disabled and blocked ones).",
          {"deletion_requested": {"type": "boolean"},
           "rationale": {"type": "string", "description": "Short explanation of how you reached the decision."},
           "records": {"type": "array", "items": {"type": "object", "properties": {
               "system_id": S, "resource_id": S,
               "state_at_audit": {"type": "string", "enum": sorted(STATE_TABLE)},
               "evidence_id": {"type": "string", "description": "evidence file_id / activity_id / audit_id proving the state"},
               "note": S}, "required": ["system_id", "resource_id", "state_at_audit"]}}},
          ["deletion_requested", "rationale"]),
]
TOOLS = types.Tool(function_declarations=TOOL_DECLARATIONS)

# ----------------------------------------------------------------------------------------------
# THE BRAIN: Gemini wrapper (system prompt, retries, thinking, token accounting)
# ----------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the Ghost-Data Deletion Audit Agent for Northstar Shop. For one request at a time you must prove
whether a person's personal data is truly gone at the case's `audit_at`, using ONLY tools and evidence at or before audit_at.

WORKFLOW (think briefly in plain text before every tool call: what you know, what you need, why):
1. DID THEY ASK FOR DELETION? Read request_text (any language). Deletion/erasure/closure of account or personal data = yes.
   Requests only to export/copy data, correct data, update preferences, unsubscribe, ask a question, or that the privacy
   team classified as "non-deletion" (see CLASSIFY_REQUEST activity) = NO. If NO: call submit_audit(deletion_requested=false) immediately.
2. resolve_identity(email) -> user_id plus ALL historical emails/phones/hashes. The supplied email may be old.
3. find_user_footprint(user_id) -> every resource that exposes or is linked to the user. Then use get_resource_timeline,
   query_sql_snapshot (compare primary vs replica tables), read_visual_evidence and search_evidence to settle doubtful records.
   Resources the activity stream shows being deleted/anonymized/redacted for this user may have NO identifiers in any evidence: for those
   (see timing_candidates) call check_absence to find the latest snapshot proving absence, and use that snapshot's file_id as evidence_id.
4. Decide each resource's state AT audit_at. Rules of evidence:
   - `reported_status` and activity `reported_result` (SUCCESS/FAILED/PARTIAL) are CLAIMS, not proof. Observed record contents beat claims.
   - The latest observation at or before audit_at wins; a later sync/restore/import can bring back data that an earlier delete removed.
   - A record that exists only as a masked/null shell (404, no identifiers) is absent/clean. Replicas/failover copies count as copies.
   - An enabled connector/data flow or a sync/import that can copy or recreate the user's data = ACTIVE_DATA_FLOW even if the primary record is gone.
   - A backup whose restore_allowed is true (or restore not blocked) can resurrect the user = RESTORE_RISK.
5. Apply deletion_rules.csv per category: DELETE (profile, secondary identifier, search index, cache, export, marketing profile, sole-owner
   attachment) -> REMAINING_RECORD if still present; REDACT (support message with shared context, email delivery log with an email,
   account tombstone keeping only user_id) -> EXPOSED_PII_REDACT if direct identifiers (name/email/phone/email_hash) remain;
   ANONYMIZE (analytics profile, product telemetry still attributable) -> EXPOSED_PII_ANONYMIZE; RETAIN (security audit log inside its
   retention period, active legal hold, backup archive with restore blocked) -> OK_RETAINED / OK_RESTORE_BLOCKED, never delete these.
   Retained records may keep only the permitted stable reference (user_id), no direct identifiers.
6. submit_audit with one record per relevant resource: system_id, resource_id, state_at_audit, evidence_id.
   Use OK_* states for resources already in the required end state (e.g. OK_ABSENT for a deleted account, OK_RETAINED for audit records).
   evidence_id MUST be a real evidence file_id, activity_id or audit_id that proves the state you assert (latest relevant at audit_at).
   Include the user's account record, tombstone, retained audit/legal records, data flows and backups whenever they exist for this user.

Never invent resource ids, systems or evidence ids. Ignore unrelated users' records (decoys). If submit_audit returns errors, fix them and resubmit.
Be efficient: typically 3-7 tool calls. Do not call tools after your submission is accepted."""


class Brain:
    def __init__(self, client, usage: Usage):
        self.client, self.usage = client, usage
        try:
            self.thinking = types.ThinkingConfig(include_thoughts=True, thinking_level=THINKING_LEVEL)
        except Exception:
            self.thinking = types.ThinkingConfig(include_thoughts=True)

    def _config(self, force_submit: bool = False) -> types.GenerateContentConfig:
        kw: dict = {"system_instruction": SYSTEM_PROMPT, "tools": [TOOLS], "thinking_config": self.thinking,
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
        if force_submit:
            kw["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["submit_audit"]))
        return types.GenerateContentConfig(**kw)

    def think(self, contents: list[types.Content], force_submit: bool = False):
        delay = 2.0
        for attempt in range(7):
            try:
                resp = self.client.models.generate_content(model=MODEL, contents=contents, config=self._config(force_submit))
                self.usage.add(resp)
                return resp
            except Exception as e:
                msg = str(e)
                if "thinking" in msg.lower() and "include_thoughts" in msg.lower():
                    self.thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
                transient = any(t in msg for t in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE", "timed out"))
                if attempt == 6 or not transient:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)


# ----------------------------------------------------------------------------------------------
# THE ENGINE: Think -> Act -> Observe loop
# ----------------------------------------------------------------------------------------------
def fallback_submission(tb: ToolBox, case: dict) -> dict:
    """Last resort when the Brain never submits: deterministic best effort from raw hints."""
    recs = []
    res = tb.run("resolve_identity", {"email": case["email"]})
    uid = (res.get("matches") or [{}])[0].get("user_id")
    if not uid:
        return {"deletion_requested": True, "rationale": "fallback: identity not resolved", "records": []}
    fp = tb.run("find_user_footprint", {"user_id": uid})
    for r in fp.get("resources", []):
        if not r.get("latest_observations"):
            continue
        ob, pre = r["latest_observations"][-1], prefix_of(r["resource_id"])
        h, sysid = ob["hint"], r.get("system_id", ob.get("system") or "")
        if pre == "data_flow":
            st = "ACTIVE_DATA_FLOW" if str(h.get("enabled")).lower() in ("1", "true") else "OK_FLOW_DISABLED"
        elif pre == "backup":
            st = "RESTORE_RISK" if str(h.get("restore_allowed")).lower() in ("1", "true") else "OK_RESTORE_BLOCKED"
        elif pre in ("audit_record", "legal_hold"):
            st = "OK_RETAINED"
        elif h["looks_absent"] and not h["pii_fields_present"]:
            st = "OK_ABSENT"
        elif pre in ("analytics",):
            st = "EXPOSED_PII_ANONYMIZE" if h["pii_fields_present"] else "OK_ANONYMIZED"
        elif pre in ("support", "email_delivery", "tombstone"):
            st = "EXPOSED_PII_REDACT" if h["pii_fields_present"] else "OK_REDACTED"
        else:
            st = "REMAINING_RECORD" if h["pii_fields_present"] else "OK_ABSENT"
        recs.append({"system_id": sysid, "resource_id": r["resource_id"], "state_at_audit": st, "evidence_id": ob["evidence_id"]})
    # the user's account may be visible only through deletion-worker activity: take the closest one and test it for absence
    cands = [t for t in fp.get("timing_candidates", []) if prefix_of(t["resource_id"]) == "account"]
    if cands and not any(prefix_of(r["resource_id"]) == "account" for r in recs):
        rid = cands[0]["resource_id"]
        rows = tb.run("check_absence", {"resource_id": rid}).get("covering_evidence_best_first", [])
        if rows:
            recs.append({"system_id": "account_service", "resource_id": rid, "evidence_id": rows[0]["evidence_id"],
                         "state_at_audit": "REMAINING_RECORD" if rows[0]["resource_present"] else "OK_ABSENT"})
    return {"deletion_requested": True, "rationale": "fallback heuristic (agent did not submit)", "records": recs}


class AgentLoop:
    def __init__(self, shared: Shared, brain: Brain):
        self.s, self.brain = shared, brain

    def _opening_prompt(self, case: dict) -> str:
        shots = ""
        if self.s.ltm:
            hits = self.s.ltm.search(case["request_text"], k=FEW_SHOT_K, sources={"gold"}, exclude_case=case["case_id"])
            if hits:
                shots = "\n\nSimilar SOLVED cases from long-term memory (verified answers; resource ids differ per case):\n" + "\n".join(
                    f"- request: {m['text'][:220]}\n  -> status={m['payload']['status']} key_steps: {m['payload']['key_steps'][:500]}" for m in hits)
        return (f"CASE {case['case_id']}\nemail supplied: {case['email']}\nrequested_at: {case['requested_at']}\n"
                f"reported_status (a CLAIM): {case['reported_status']} at {case['reported_at']}\naudit_at: {case['audit_at']}\n"
                f"request_text: \"\"\"{case['request_text']}\"\"\"{shots}\n\nBegin the audit.")

    def run(self, case: dict) -> dict:
        tb = ToolBox(self.s, case)
        stm = ShortTermMemory()
        lines = [f"{'=' * 100}", f"CASE {case['case_id']}  [{RunLog.now()}]",
                 f"email={case['email']} reported_status={case['reported_status']} audit_at={case['audit_at']}",
                 f"request_text: {case['request_text']}"]
        stm.add_user_text(self._opening_prompt(case))
        forced = False
        for it in range(1, MAX_ITERATIONS + 2):
            force = it == MAX_ITERATIONS + 1  # final iteration: force submit_audit
            if force:
                forced = True
                lines.append("-- iteration limit reached: forcing submit_audit")
            lines.append(f"\n--- iteration {it} ---")
            try:
                resp = self.brain.think(stm.contents, force_submit=force)
            except Exception as e:
                lines.append(f"BRAIN ERROR: {type(e).__name__}: {str(e)[:300]}")
                break
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None or not cand.content.parts:
                lines.append("BRAIN returned no content; nudging")
                stm.add_user_text("Continue: call the next tool, or submit_audit if you have enough evidence.")
                continue
            stm.add_model(cand.content)
            calls = []
            for p in cand.content.parts:                       # ---- THINK
                if p.function_call:
                    calls.append(p.function_call)
                elif p.text:
                    lines.append(("THOUGHT (summary): " if getattr(p, "thought", False) else "THINK: ") + p.text.strip())
            if not calls:
                lines.append("(no tool call) nudging the Brain to continue")
                stm.add_user_text("You must continue with a tool call or finish with submit_audit.")
                continue
            results = []
            for fc in calls:                                   # ---- ACT
                args = dict(fc.args or {})
                lines.append(f"ACT: {fc.name}({clip(json.dumps(args, ensure_ascii=False), 700)})")
                out = tb.run(fc.name, args)
                text = json.dumps(out, ensure_ascii=False, default=str)
                lines.append(f"OBSERVE: {clip(text, 1500)}")  # ---- OBSERVE
                results.append(types.Part.from_function_response(
                    name=fc.name, response={"result": json.loads(clip(text, MAX_TOOL_CHARS)) if len(text) <= MAX_TOOL_CHARS else clip(text, MAX_TOOL_CHARS)}))
            if tb.submission:
                break
            stm.add_tool_results(results)
            stm.compact()
        used_fallback = False
        if not tb.submission:
            used_fallback = True
            lines.append("!! Brain never produced an accepted submission -> deterministic fallback")
            tb.submission = fallback_submission(tb, case)
        sub = tb.submission
        status, steps = build_plan(sub["deletion_requested"], sub["records"])
        lines += ["", f"FINAL DECISION: {status}  (forced={forced}, fallback={used_fallback})",
                  f"RATIONALE: {sub['rationale']}", "RECORD STATES:"]
        lines += [f"  - {r['system_id']}/{r['resource_id']}: {r['state_at_audit']}  evidence={r['evidence_id'] or '-'}  {r.get('note', '')}"
                  for r in sub["records"]]
        lines.append("STEPS: " + (json.dumps(steps, ensure_ascii=False) if steps else "[]"))
        self.s.log.write("\n".join(lines))
        return {"case_id": case["case_id"], "deletion_status": status, "steps": steps,
                "rationale": sub["rationale"], "summary": summarize_steps(steps)}


# ----------------------------------------------------------------------------------------------
# Orchestration, evaluation, entry point
# ----------------------------------------------------------------------------------------------
def write_output(results: dict[str, dict], cases: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "deletion_status", "steps_json"])
        for c in sorted(cases, key=lambda c: case_sort_key(c["case_id"])):
            r = results[c["case_id"]]
            w.writerow([c["case_id"], r["deletion_status"], json.dumps(r["steps"], ensure_ascii=False, separators=(",", ":"))])


def evaluate(results: dict[str, dict], log: RunLog):
    gold = {r["case_id"]: r for r in read_csv(DATASET / "sample_output.csv")}
    n = ok = seq = 0
    tp = fp = fn = 0
    for cid, r in results.items():
        g = gold.get(cid)
        if not g:
            continue
        n += 1
        ok += g["deletion_status"] == r["deletion_status"]
        gs = json.loads(g["steps_json"])
        key = lambda s: (s["phase"], s["operation"], s["system_id"], s["resource_id"], s["state"])
        a, b = {key(s) for s in gs}, {key(s) for s in r["steps"]}
        tp, fp, fn = tp + len(a & b), fp + len(b - a), fn + len(a - b)
        seq += [key(s) for s in gs] == [key(s) for s in r["steps"]]
    p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    msg = (f"[eval] cases={n} status_accuracy={ok / max(n, 1):.3f} step_precision={p:.3f} step_recall={rc:.3f} "
           f"step_f1={2 * p * rc / max(p + rc, 1e-9):.3f} exact_sequence_match={seq / max(n, 1):.3f}")
    print(msg)
    log.write(msg)


def main():
    ap = argparse.ArgumentParser(description="Ghost-data deletion audit agent")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    ap.add_argument("--cases", type=str, default="", help="comma-separated case_ids")
    ap.add_argument("--eval-samples", action="store_true", help="run on sample requests and score vs sample_output.csv")
    ap.add_argument("--fresh", action="store_true", help="ignore cached case results")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY (environment variable or .env file) before running.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log, usage = RunLog(LOG_TXT), Usage()
    log.write(f"\n{'#' * 100}\nRUN START {RunLog.now()}  model={MODEL}  mode={'samples' if args.eval_samples else 'full'}\n{'#' * 100}")
    client = genai.Client()

    req_file = "sample_deletion_requests.csv" if args.eval_samples else "deletion_requests.csv"
    cases = read_csv(DATASET / req_file)
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c["case_id"] in want]
    if args.limit:
        cases = cases[: args.limit]

    print("Indexing evidence (first run transcribes PDFs/images with Gemini)...")
    shared = Shared(client, usage, log)
    shared.ev.build(client, log)
    log.write(f"[index] {len(shared.ev.obs)} observations from {len(shared.ev.manifest)} evidence files; "
              f"{len(shared.activity.rows)} activity rows; {len(shared.ids.cur)} users")
    shared.ltm = LongTermMemory(client, usage, log)
    seed_long_term_memory(shared.ltm)
    log.write(f"[memory] long-term memory holds {len(shared.ltm.meta)} entries")

    cache: dict[str, dict] = {}
    if RESULTS_CACHE.exists() and not args.fresh and not args.eval_samples:
        for line in RESULTS_CACHE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                cache[r["case_id"]] = r
            except ValueError:
                pass
    todo = [c for c in cases if c["case_id"] not in cache]
    print(f"{len(cases)} cases, {len(cache)} cached, {len(todo)} to run with {args.workers} workers")

    brain = Brain(client, usage)
    loop = AgentLoop(shared, brain)
    results = {c["case_id"]: cache[c["case_id"]] for c in cases if c["case_id"] in cache}
    cache_lock = threading.Lock()

    def work(case):
        try:
            return loop.run(case)
        except Exception:
            err = traceback.format_exc()
            log.write(f"CASE {case['case_id']} CRASHED:\n{err}")
            return {"case_id": case["case_id"], "deletion_status": "INCOMPLETE", "steps": [], "rationale": "crashed", "summary": ""}

    done = 0
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["case_id"]] = r
            done += 1
            if not args.eval_samples and r["rationale"] != "crashed":
                with cache_lock, open(RESULTS_CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{done}/{len(todo)}] {r['case_id']} -> {r['deletion_status']} ({len(r['steps'])} steps)")

    if args.eval_samples:
        evaluate(results, log)
    else:
        write_output(results, cases, OUTPUT_CSV)
        print(f"Wrote {OUTPUT_CSV} ({len(results)} rows)")
        # persist this run's outcomes in long-term memory (tagged unverified) for future runs
        shared.ltm.add([{"source": "agent", "case_id": r["case_id"], "text": next(c["request_text"] for c in cases if c["case_id"] == r["case_id"]),
                         "payload": {"status": r["deletion_status"], "key_steps": r["summary"], "n_steps": len(r["steps"])}}
                        for r in results.values() if r["rationale"] != "crashed"][:600])
    log.write(f"RUN END {RunLog.now()}  {usage.summary()}")
    print(usage.summary())


if __name__ == "__main__":
    main()
