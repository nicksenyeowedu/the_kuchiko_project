"""
Build Report Tracker for Kuchiko Project

Tracks token usage AND time taken during the build phase:
- LLM chat completion tokens (input/output) via NVIDIA NIM
- Embedding API tokens via NVIDIA NIM
- Local embedding calls via sentence-transformers (no cost, tracked for stats)
- Per-call timing and per-step timing

Usage:
    from token_tracker import tracker

    # Track a build step:
    tracker.start_step("Knowledge Graph Creation")
    ...
    tracker.end_step("Knowledge Graph Creation")

    # After an LLM chat completion call:
    tracker.log_chat_completion(response, caller="createKG.ask_nim", prompt_preview="...")

    # After an embedding API call:
    tracker.log_embedding(texts, caller="build_embeddings.get_embeddings_batch")

    # After local embedding (sentence-transformers):
    tracker.log_local_embedding(text, caller="createKG.get_embedding")

    # Write summary report:
    tracker.write_report()

Output:
    - logs/token_usage.jsonl  (per-call log, append-only)
    - logs/build_report.txt   (human-readable summary)
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import List, Optional
from collections import OrderedDict


class TokenTracker:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "token_usage.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        self._lock = threading.Lock()

        # Running totals
        self.total_chat_input_tokens = 0
        self.total_chat_output_tokens = 0
        self.total_chat_calls = 0

        self.total_embedding_tokens = 0
        self.total_embedding_calls = 0
        self.total_texts_embedded = 0

        self.total_local_embedding_calls = 0
        self.total_local_texts_embedded = 0

        # Per-caller breakdown
        self.caller_stats = {}

        # Timing
        self.start_time = datetime.now()
        self._start_monotonic = time.monotonic()

        # Step timing (ordered to preserve sequence)
        self.steps = OrderedDict()  # name -> {start, end, duration, chat_calls, emb_calls, ...}
        self._active_step = None

    # ---- Step timing ----

    def start_step(self, name: str):
        """Mark the start of a build step (e.g. 'PDF Extraction', 'Knowledge Graph Creation')."""
        with self._lock:
            self.steps[name] = {
                "start": datetime.now().isoformat(),
                "start_mono": time.monotonic(),
                "end": None,
                "duration_seconds": None,
                "chat_calls": 0,
                "chat_input_tokens": 0,
                "chat_output_tokens": 0,
                "embedding_calls": 0,
                "embedding_tokens": 0,
                "local_embedding_calls": 0,
            }
            self._active_step = name

    def end_step(self, name: str = None):
        """Mark the end of a build step. If name is None, ends the active step."""
        with self._lock:
            step_name = name or self._active_step
            if step_name and step_name in self.steps:
                step = self.steps[step_name]
                step["end"] = datetime.now().isoformat()
                step["duration_seconds"] = round(time.monotonic() - step["start_mono"], 1)
                if self._active_step == step_name:
                    self._active_step = None

    def _update_active_step(self, call_type: str, input_tokens: int = 0,
                            output_tokens: int = 0, embedding_tokens: int = 0):
        """Update the active step's counters."""
        if self._active_step and self._active_step in self.steps:
            step = self.steps[self._active_step]
            if call_type == "chat":
                step["chat_calls"] += 1
                step["chat_input_tokens"] += input_tokens
                step["chat_output_tokens"] += output_tokens
            elif call_type == "embedding":
                step["embedding_calls"] += 1
                step["embedding_tokens"] += embedding_tokens
            elif call_type == "local_embedding":
                step["local_embedding_calls"] += 1

    # ---- Caller stats ----

    def _update_caller(self, caller: str, input_tokens: int = 0, output_tokens: int = 0,
                       embedding_tokens: int = 0, texts_count: int = 0, call_type: str = "chat",
                       elapsed_ms: float = 0):
        if caller not in self.caller_stats:
            self.caller_stats[caller] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "embedding_tokens": 0,
                "texts_embedded": 0,
                "total_time_ms": 0,
                "type": call_type
            }
        stats = self.caller_stats[caller]
        stats["calls"] += 1
        stats["input_tokens"] += input_tokens
        stats["output_tokens"] += output_tokens
        stats["embedding_tokens"] += embedding_tokens
        stats["texts_embedded"] += texts_count
        stats["total_time_ms"] += elapsed_ms

    def _append_log(self, entry: dict):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---- Logging methods ----

    def log_chat_completion(self, response, caller: str = "unknown",
                            prompt_preview: str = "", elapsed_ms: float = 0):
        """Log an OpenAI-compatible chat completion response.

        Args:
            response: The response object from nim_client.chat.completions.create()
            caller: Identifier for where this call originated
            prompt_preview: First ~100 chars of the prompt for debugging
            elapsed_ms: Time taken for this API call in milliseconds
        """
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens)

        with self._lock:
            self.total_chat_input_tokens += input_tokens
            self.total_chat_output_tokens += output_tokens
            self.total_chat_calls += 1
            self._update_caller(caller, input_tokens=input_tokens, output_tokens=output_tokens,
                                call_type="chat", elapsed_ms=elapsed_ms)
            self._update_active_step("chat", input_tokens=input_tokens, output_tokens=output_tokens)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "chat_completion",
            "caller": caller,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "elapsed_ms": round(elapsed_ms, 1),
            "prompt_preview": prompt_preview[:150]
        }
        self._append_log(entry)

    def log_embedding(self, texts: list, response=None, caller: str = "unknown",
                      elapsed_ms: float = 0):
        """Log an NVIDIA NIM embedding API call.

        Args:
            texts: The list of texts that were embedded
            response: The response object from nim_client.embeddings.create()
            caller: Identifier for where this call originated
            elapsed_ms: Time taken for this API call in milliseconds
        """
        num_texts = len(texts) if texts else 0
        embedding_tokens = 0

        usage = getattr(response, "usage", None) if response else None
        if usage:
            embedding_tokens = getattr(usage, "total_tokens", 0) or 0
            if embedding_tokens == 0:
                embedding_tokens = getattr(usage, "prompt_tokens", 0) or 0

        # Estimate tokens if API didn't return usage (rough: ~1 token per 4 chars)
        if embedding_tokens == 0 and texts:
            embedding_tokens = sum(len(t) // 4 for t in texts)

        with self._lock:
            self.total_embedding_tokens += embedding_tokens
            self.total_embedding_calls += 1
            self.total_texts_embedded += num_texts
            self._update_caller(caller, embedding_tokens=embedding_tokens, texts_count=num_texts,
                                call_type="embedding", elapsed_ms=elapsed_ms)
            self._update_active_step("embedding", embedding_tokens=embedding_tokens)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "embedding_api",
            "caller": caller,
            "num_texts": num_texts,
            "embedding_tokens": embedding_tokens,
            "elapsed_ms": round(elapsed_ms, 1),
            "text_preview": texts[0][:100] if texts else ""
        }
        self._append_log(entry)

    def log_local_embedding(self, texts_count: int = 1, caller: str = "unknown",
                            elapsed_ms: float = 0):
        """Log a local sentence-transformers embedding call (no API cost).

        Args:
            texts_count: Number of texts embedded
            caller: Identifier for where this call originated
            elapsed_ms: Time taken in milliseconds
        """
        with self._lock:
            self.total_local_embedding_calls += 1
            self.total_local_texts_embedded += texts_count
            self._update_caller(caller, texts_count=texts_count, call_type="local_embedding",
                                elapsed_ms=elapsed_ms)
            self._update_active_step("local_embedding")

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "local_embedding",
            "caller": caller,
            "num_texts": texts_count,
            "elapsed_ms": round(elapsed_ms, 1)
        }
        self._append_log(entry)

    # ---- State persistence (for unified reports across processes) ----

    def save_state(self, filename: Optional[str] = None):
        """Save tracker state to a JSON file so a later process can resume accumulation."""
        if filename is None:
            filename = os.path.join(self.log_dir, "tracker_state.json")
        state = {
            "total_chat_input_tokens": self.total_chat_input_tokens,
            "total_chat_output_tokens": self.total_chat_output_tokens,
            "total_chat_calls": self.total_chat_calls,
            "total_embedding_tokens": self.total_embedding_tokens,
            "total_embedding_calls": self.total_embedding_calls,
            "total_texts_embedded": self.total_texts_embedded,
            "total_local_embedding_calls": self.total_local_embedding_calls,
            "total_local_texts_embedded": self.total_local_texts_embedded,
            "caller_stats": self.caller_stats,
            "steps": {k: {kk: vv for kk, vv in v.items() if kk != "start_mono"}
                      for k, v in self.steps.items()},
            "start_time": self.start_time.isoformat(),
            "_start_monotonic_elapsed": time.monotonic() - self._start_monotonic,
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_state(self, filename: Optional[str] = None):
        """Load previously saved tracker state and merge it into this instance."""
        if filename is None:
            filename = os.path.join(self.log_dir, "tracker_state.json")
        if not os.path.exists(filename):
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return

        with self._lock:
            self.total_chat_input_tokens += state.get("total_chat_input_tokens", 0)
            self.total_chat_output_tokens += state.get("total_chat_output_tokens", 0)
            self.total_chat_calls += state.get("total_chat_calls", 0)
            self.total_embedding_tokens += state.get("total_embedding_tokens", 0)
            self.total_embedding_calls += state.get("total_embedding_calls", 0)
            self.total_texts_embedded += state.get("total_texts_embedded", 0)
            self.total_local_embedding_calls += state.get("total_local_embedding_calls", 0)
            self.total_local_texts_embedded += state.get("total_local_texts_embedded", 0)

            # Merge caller stats
            for caller, stats in state.get("caller_stats", {}).items():
                if caller not in self.caller_stats:
                    self.caller_stats[caller] = stats
                else:
                    existing = self.caller_stats[caller]
                    existing["calls"] += stats.get("calls", 0)
                    existing["input_tokens"] += stats.get("input_tokens", 0)
                    existing["output_tokens"] += stats.get("output_tokens", 0)
                    existing["embedding_tokens"] += stats.get("embedding_tokens", 0)
                    existing["texts_embedded"] += stats.get("texts_embedded", 0)
                    existing["total_time_ms"] += stats.get("total_time_ms", 0)

            # Prepend prior steps (they ran first)
            prior_steps = OrderedDict()
            for k, v in state.get("steps", {}).items():
                prior_steps[k] = v
            for k, v in self.steps.items():
                prior_steps[k] = v
            self.steps = prior_steps

            # Adjust start time to the earlier run
            prior_start = state.get("start_time")
            if prior_start:
                prior_dt = datetime.fromisoformat(prior_start)
                if prior_dt < self.start_time:
                    prior_elapsed = state.get("_start_monotonic_elapsed", 0)
                    self.start_time = prior_dt
                    self._start_monotonic = time.monotonic() - prior_elapsed

    # ---- Summary & Report ----

    def get_summary(self) -> dict:
        """Return a summary dict of all token usage and timing."""
        elapsed = time.monotonic() - self._start_monotonic
        with self._lock:
            return {
                "elapsed_seconds": round(elapsed, 1),
                "chat_completions": {
                    "total_calls": self.total_chat_calls,
                    "total_input_tokens": self.total_chat_input_tokens,
                    "total_output_tokens": self.total_chat_output_tokens,
                    "total_tokens": self.total_chat_input_tokens + self.total_chat_output_tokens
                },
                "embedding_api": {
                    "total_calls": self.total_embedding_calls,
                    "total_texts_embedded": self.total_texts_embedded,
                    "total_tokens": self.total_embedding_tokens
                },
                "local_embeddings": {
                    "total_calls": self.total_local_embedding_calls,
                    "total_texts_embedded": self.total_local_texts_embedded
                },
                "grand_total_api_tokens": (
                    self.total_chat_input_tokens +
                    self.total_chat_output_tokens +
                    self.total_embedding_tokens
                ),
                "caller_breakdown": dict(self.caller_stats),
                "steps": dict(self.steps)
            }

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format seconds into a human-friendly string."""
        if seconds is None:
            return "in progress..."
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(seconds, 60)
        if m < 60:
            return f"{int(m)}m {s:.0f}s"
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"

    def write_report(self, filename: Optional[str] = None):
        """Write a human-readable build report with token usage and timing."""
        if filename is None:
            filename = os.path.join(self.log_dir, "build_report.txt")

        summary = self.get_summary()
        elapsed = summary["elapsed_seconds"]
        chat = summary["chat_completions"]
        emb = summary["embedding_api"]
        local = summary["local_embeddings"]

        lines = []
        lines.append("=" * 70)
        lines.append("KUCHIKO BUILD REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated:      {datetime.now().isoformat()}")
        lines.append(f"Total duration: {self._fmt_duration(elapsed)}")
        lines.append("")

        # ---- Step timing ----
        if summary["steps"]:
            lines.append("-" * 70)
            lines.append("BUILD STEPS")
            lines.append("-" * 70)
            for step_name, step in summary["steps"].items():
                dur = step.get("duration_seconds")
                dur_str = self._fmt_duration(dur) if dur is not None else "in progress..."
                lines.append(f"  {step_name:<40} {dur_str:>12}")

                # Show token breakdown per step
                step_tokens = (step.get("chat_input_tokens", 0) +
                               step.get("chat_output_tokens", 0) +
                               step.get("embedding_tokens", 0))
                details = []
                if step.get("chat_calls"):
                    details.append(f"{step['chat_calls']} LLM calls, "
                                   f"{step['chat_input_tokens']+step['chat_output_tokens']:,} tokens")
                if step.get("embedding_calls"):
                    details.append(f"{step['embedding_calls']} embed calls, "
                                   f"{step['embedding_tokens']:,} tokens")
                if step.get("local_embedding_calls"):
                    details.append(f"{step['local_embedding_calls']} local embeds")
                if details:
                    lines.append(f"    {' | '.join(details)}")
            lines.append("")

        # ---- Token usage ----
        lines.append("-" * 70)
        lines.append("LLM CHAT COMPLETIONS (NVIDIA NIM)")
        lines.append("-" * 70)
        lines.append(f"  Total API calls:    {chat['total_calls']}")
        lines.append(f"  Input tokens:       {chat['total_input_tokens']:,}")
        lines.append(f"  Output tokens:      {chat['total_output_tokens']:,}")
        lines.append(f"  Total tokens:       {chat['total_tokens']:,}")
        if chat["total_calls"] > 0:
            avg_time = sum(s.get("total_time_ms", 0) for s in summary["caller_breakdown"].values()
                          if s["type"] == "chat") / chat["total_calls"]
            lines.append(f"  Avg time per call:  {avg_time:.0f}ms")
        lines.append("")

        lines.append("-" * 70)
        lines.append("EMBEDDING API (NVIDIA NIM)")
        lines.append("-" * 70)
        lines.append(f"  Total API calls:    {emb['total_calls']}")
        lines.append(f"  Texts embedded:     {emb['total_texts_embedded']:,}")
        lines.append(f"  Total tokens:       {emb['total_tokens']:,}")
        if emb["total_calls"] > 0:
            avg_time = sum(s.get("total_time_ms", 0) for s in summary["caller_breakdown"].values()
                          if s["type"] == "embedding") / emb["total_calls"]
            lines.append(f"  Avg time per call:  {avg_time:.0f}ms")
        lines.append("")

        lines.append("-" * 70)
        lines.append("LOCAL EMBEDDINGS (sentence-transformers, no API cost)")
        lines.append("-" * 70)
        lines.append(f"  Total calls:        {local['total_calls']}")
        lines.append(f"  Texts embedded:     {local['total_texts_embedded']:,}")
        if local["total_calls"] > 0:
            avg_time = sum(s.get("total_time_ms", 0) for s in summary["caller_breakdown"].values()
                          if s["type"] == "local_embedding") / local["total_calls"]
            lines.append(f"  Avg time per call:  {avg_time:.0f}ms")
        lines.append("")

        lines.append("-" * 70)
        lines.append("GRAND TOTAL")
        lines.append("-" * 70)
        lines.append(f"  Total API tokens:   {summary['grand_total_api_tokens']:,}")
        lines.append(f"  Total build time:   {self._fmt_duration(elapsed)}")
        lines.append("")

        # ---- Per-caller breakdown ----
        lines.append("-" * 70)
        lines.append("PER-CALLER BREAKDOWN")
        lines.append("-" * 70)
        for caller, stats in sorted(summary["caller_breakdown"].items()):
            call_type = stats["type"]
            calls = stats["calls"]
            time_str = self._fmt_duration(stats["total_time_ms"] / 1000)
            if call_type == "chat":
                tokens = stats["input_tokens"] + stats["output_tokens"]
                lines.append(f"  {caller:<40} {calls:>4} calls  {tokens:>8,} tok  {time_str:>10}")
            elif call_type == "embedding":
                lines.append(f"  {caller:<40} {calls:>4} calls  {stats['embedding_tokens']:>8,} tok  {time_str:>10}")
            else:
                lines.append(f"  {caller:<40} {calls:>4} calls  {'local':>8}      {time_str:>10}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_text)
        except Exception:
            pass
        return report_text


# Global singleton tracker instance
tracker = TokenTracker()
