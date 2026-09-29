"""MemoryOS core service layer.

A lightweight, dependency-friendly memory engine used by the dashboard and
future API/agent integrations. It keeps persistence separate from Streamlit.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


class MemoryEngine:
    def __init__(self, data_dir: str = ".",
                 drift_threshold: float = 0.45,
                 stale_threshold: float = 0.30):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold
        self.stale_threshold = stale_threshold
        self.memory_file = self.data_dir / "memoryos_data.json"
        self.drift_file = self.data_dir / "drift_history.json"
        self.heal_file = self.data_dir / "heal_log.json"

    def _read(self, path: Path, default: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    def _write(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(path)

    def memories(self) -> List[Dict[str, Any]]:
        raw = self._read(self.memory_file, [])
        if isinstance(raw, dict):
            raw = raw.get("memories") or raw.get("final_memory") or raw.get("data") or []
        result = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str):
                result.append({"content": item, "importance": 1.0,
                               "access_count": 0, "is_stale": False})
            elif isinstance(item, dict):
                text = item.get("content") or item.get("input") or item.get("text")
                if text:
                    result.append({
                        "content": str(text),
                        "importance": float(item.get("importance", 1.0)),
                        "access_count": int(item.get("access_count", 0)),
                        "is_stale": bool(item.get("is_stale", False)),
                        "created_at": item.get("created_at"),
                    })
        return result

    def save_memories(self, memories: List[Dict[str, Any]]) -> None:
        self._write(self.memory_file, memories)

    def add(self, content: str, importance: float = 1.0) -> Dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty.")
        item = {
            "content": content,
            "importance": round(float(np.clip(importance, 0.0, 1.0)), 3),
            "access_count": 0,
            "is_stale": False,
            "created_at": datetime.now().isoformat(),
        }
        memories = self.memories()
        memories.append(item)
        self.save_memories(memories)
        return item

    @staticmethod
    def _fallback_similarity(a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / math.sqrt(len(wa) * len(wb))

    def drift(self, baseline: List[str], current: List[str]) -> Dict[str, Any]:
        baseline = [x.strip() for x in baseline if x and x.strip()]
        current = [x.strip() for x in current if x and x.strip()]
        if not baseline or not current:
            raise ValueError("Baseline and current inputs are required.")

        if TfidfVectorizer is not None:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=500, ngram_range=(1, 2), stop_words="english"
                )
                matrix = vectorizer.fit_transform(baseline + current)
                base = matrix[:len(baseline)].mean(axis=0)
                cur = matrix[len(baseline):].mean(axis=0)
                similarity = float(cosine_similarity(base, cur)[0][0])
            except Exception:
                similarity = float(np.mean([
                    self._fallback_similarity(baseline[0], text)
                    for text in current
                ]))
        else:
            similarity = float(np.mean([
                self._fallback_similarity(baseline[0], text)
                for text in current
            ]))

        score = round(float(np.clip(1.0 - similarity, 0.0, 1.0)), 4)
        if score < 0.30:
            status, emoji = "HEALTHY", "🟢"
        elif score < 0.50:
            status, emoji = "WARNING", "🟡"
        elif score < 0.70:
            status, emoji = "CRITICAL", "🔴"
        else:
            status, emoji = "DANGER", "🚨"

        result = {
            "score": score,
            "status": status,
            "emoji": emoji,
            "heal_needed": score >= self.drift_threshold,
            "timestamp": datetime.now().isoformat(),
            "baseline_count": len(baseline),
            "current_count": len(current),
        }
        history = self._read(self.drift_file, [])
        if not isinstance(history, list):
            history = []
        result["session"] = len(history) + 1
        history.append(result)
        self._write(self.drift_file, history)
        return result

    def drift_history(self) -> List[Dict[str, Any]]:
        value = self._read(self.drift_file, [])
        return value if isinstance(value, list) else []

    def heal(self, drift_score: float) -> Dict[str, Any]:
        memories = self.memories()
        before = len(memories)
        session_count = max(1, len(self.drift_history()))

        scored = []
        for memory in memories:
            importance = float(memory.get("importance", 1.0))
            accesses = int(memory.get("access_count", 0))
            relevance = importance * math.exp(-0.1 * session_count) + min(accesses, 10) * 0.01
            memory["relevance"] = round(relevance, 4)
            memory["is_stale"] = relevance < self.stale_threshold
            scored.append((relevance, memory))

        active = [m for _, m in sorted(scored, key=lambda x: x[0], reverse=True)
                  if not m["is_stale"]]
        # Preserve a useful working set even when everything has decayed.
        if not active and scored:
            active = [m for _, m in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
            for m in active:
                m["is_stale"] = False

        retained = active[:50]
        for memory in retained[:3]:
            memory["access_count"] = int(memory.get("access_count", 0)) + 1

        self.save_memories(retained)
        entry = {
            "session": session_count,
            "timestamp": datetime.now().isoformat(),
            "drift_score_before": float(drift_score),
            "memories_before": before,
            "memories_pruned": max(0, before - len(retained)),
            "memories_after": len(retained),
            "reinjected": [m["content"] for m in retained[:3]],
        }
        logs = self._read(self.heal_file, [])
        if not isinstance(logs, list):
            logs = []
        logs.append(entry)
        self._write(self.heal_file, logs)
        return entry

    def heal_history(self) -> List[Dict[str, Any]]:
        value = self._read(self.heal_file, [])
        return value if isinstance(value, list) else []

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = query.strip().lower()
        memories = self.memories()
        if not query:
            return memories[:limit]
        if TfidfVectorizer is None:
            return [m for m in memories if query in m["content"].lower()][:limit]

        texts = [m["content"] for m in memories]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(texts + [query])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
            ranked = np.argsort(scores)[::-1]
            return [memories[i] | {"similarity": round(float(scores[i]), 4)}
                    for i in ranked[:limit] if scores[i] > 0]
        except Exception:
            return [m for m in memories if query in m["content"].lower()][:limit]


    def retrieve(self, query: str, limit: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """RAG-style retrieval with relevance and importance-aware ranking."""
        query = query.strip()
        if not query:
            return {"query": query, "results": []}
        memories = self.memories()
        if not memories:
            return {"query": query, "results": []}
        semantic = self.search(query, limit=max(limit * 3, 10))
        ranked = []
        for item in semantic:
            similarity = float(item.get("similarity", 0.0))
            if similarity < min_similarity:
                continue
            importance = float(item.get("importance", 1.0))
            relevance = float(item.get("relevance", importance))
            score = 0.65 * similarity + 0.25 * importance + 0.10 * min(relevance, 1.0)
            ranked.append({
                **item,
                "retrieval_score": round(score, 4),
                "similarity": round(similarity, 4),
            })
        ranked.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return {"query": query, "results": ranked[:limit]}

    def context(self, query: str, limit: int = 5) -> str:
        """Build a compact context block suitable for an agent prompt."""
        results = self.retrieve(query, limit=limit)["results"]
        if not results:
            return ""
        lines = ["Relevant memory context:"]
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. {item['content']}")
        return "\n".join(lines)

    def lifecycle(self) -> List[Dict[str, Any]]:
        """Return memory lifecycle state for dashboard/agent inspection."""
        session_count = max(1, len(self.drift_history()))
        result = []
        for item in self.memories():
            importance = float(item.get("importance", 1.0))
            accesses = int(item.get("access_count", 0))
            relevance = importance * math.exp(-0.1 * session_count) + min(accesses, 10) * 0.01
            result.append({
                **item,
                "relevance": round(relevance, 4),
                "state": "STALE" if relevance < self.stale_threshold else "ACTIVE",
            })
        return sorted(result, key=lambda x: x["relevance"], reverse=True)

    def health(self) -> Dict[str, Any]:
        history = self.drift_history()
        last = history[-1] if history else None
        score = float(last.get("score", last.get("drift_score", 0))) if last else 0.0
        status = last.get("status", "HEALTHY") if last else "HEALTHY"
        return {
            "memory_count": len(self.memories()),
            "drift_score": score,
            "status": status,
            "heal_events": len(self.heal_history()),
            "sessions": len(history),
            "threshold": self.drift_threshold,
        }
