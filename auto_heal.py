# auto_heal.py — MemoryOS Week 1, Day 3
# Auto-Heal Module — Zero downtime self recovery

import numpy as np
import json
from datetime import datetime
from drift_detector import DriftDetector, simple_embed, get_drift_status

class MemoryItem:
    def __init__(self, content: str, importance: float = 1.0):
        self.content = content
        self.importance = importance
        self.created_at = datetime.now()
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.is_stale = False

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()

    def calculate_relevance(self, sessions_elapsed: int) -> float:
        decay_rate = 0.1
        relevance = self.importance * np.exp(-decay_rate * sessions_elapsed)
        return round(relevance, 4)

    def to_dict(self):
        return {
            "content": self.content,
            "importance": self.importance,
            "access_count": self.access_count,
            "is_stale": self.is_stale
        }


class AutoHealEngine:
    def __init__(self, drift_threshold: float = 0.45):
        self.memory_store = []
        self.drift_detector = DriftDetector()
        self.drift_threshold = drift_threshold
        self.heal_log = []
        self.session_count = 0
        self.is_baseline_set = False

    def add_memory(self, content: str, importance: float = 1.0):
        item = MemoryItem(content, importance)
        self.memory_store.append(item)

    def set_baseline_from_memory(self):
        if not self.memory_store:
            print("No memories to set baseline!")
            return
        texts = [m.content for m in self.memory_store]
        self.drift_detector.set_baseline(texts)
        self.is_baseline_set = True

    def _perform_heal(self, drift_score: float):
        PRUNE_THRESHOLD = 0.3
        TOP_K = 3
        before_count = len(self.memory_store)

        print(f"\n   Step 1: Calculating relevance...")
        for memory in self.memory_store:
            relevance = memory.calculate_relevance(self.session_count)
            if relevance < PRUNE_THRESHOLD:
                memory.is_stale = True
                print(f"   Stale: '{memory.content[:40]}' (relevance: {relevance})")

        print(f"\n   Step 2: Pruning stale memories...")
        active = [m for m in self.memory_store if not m.is_stale]
        pruned = before_count - len(active)
        self.memory_store = active

        print(f"\n   Step 3: Re-injecting top {TOP_K} memories...")
        top = sorted(self.memory_store, key=lambda m: m.importance, reverse=True)[:TOP_K]
        for m in top:
            m.access()
            print(f"   Reinjected: '{m.content[:40]}'")

        entry = {
            "session": self.session_count,
            "timestamp": datetime.now().isoformat(),
            "drift_score_before": drift_score,
            "memories_before": before_count,
            "memories_pruned": pruned,
            "memories_after": len(self.memory_store),
            "reinjected": [m.content for m in top]
        }
        self.heal_log.append(entry)

        return pruned, len(self.memory_store)

    def check_and_heal(self, current_inputs: list):
        self.session_count += 1

        print(f"\n{'='*45}")
        print(f"Session {self.session_count} — Health Check")
        print(f"{'='*45}")

        result = self.drift_detector.check_drift(current_inputs)
        drift_score = result["drift_score"]
        status = result["status"]

        print(f"Drift Score: {drift_score}")
        print(f"{status['emoji']} {status['status']}: {status['message']}")
        print(f"Memory store: {len(self.memory_store)} items")

        if drift_score >= self.drift_threshold:
            print(f"\n🚨 Threshold breached! Starting Auto-Heal...")
            pruned, retained = self._perform_heal(drift_score)
            print(f"\n✅ Heal complete!")
            print(f"   Pruned: {pruned} memories")
            print(f"   Retained: {retained} memories")
        else:
            print(f"✅ Agent healthy — no heal needed")

    def show_memory_store(self):
        print(f"\nMemory Store ({len(self.memory_store)} items):")
        for i, m in enumerate(self.memory_store):
            relevance = m.calculate_relevance(self.session_count)
            tag = "STALE" if m.is_stale else "Active"
            print(f"  {i+1}. [{tag}] '{m.content[:45]}' | importance: {m.importance} | relevance: {relevance}")

    def save_logs(self):
        with open("heal_log.json", "w") as f:
            json.dump(self.heal_log, f, indent=2)
        print(f"\n💾 Heal log saved ({len(self.heal_log)} heal events)")


if __name__ == "__main__":
    print("🤖 MemoryOS — Auto Heal Engine")
    print("="*45)

    engine = AutoHealEngine(drift_threshold=0.45)

    print("\n📥 Loading memories...")
    memories = [
        ("what is machine learning", 0.9),
        ("what is deep learning", 0.85),
        ("what is neural network", 0.8),
        ("what is python programming", 0.75),
        ("what is data science", 0.7),
    ]
    for content, importance in memories:
        engine.add_memory(content, importance)
        print(f"   Added: '{content}'")

    print("\n📍 Setting baseline...")
    engine.set_baseline_from_memory()

    engine.show_memory_store()

    print("\n📌 Session 1: Normal AI questions")
    engine.check_and_heal([
        "explain artificial intelligence",
        "how does gradient descent work",
        "what is supervised learning"
    ])

    print("\n📌 Session 2: Very different topics")
    engine.check_and_heal([
        "best cricket team in world",
        "recipe for butter chicken",
        "latest bollywood movies 2026",
        "weather forecast mumbai"
    ])

    engine.show_memory_store()
    engine.save_logs()

    print("\n✅ Day 3 Complete!")
    print("   heal_log.json check karo!")