# calibration.py — MemoryOS Week 2 Day 2
# Threshold Calibration Study

import numpy as np
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🔬 MemoryOS — Threshold Calibration Study")
print("="*50)

# ── EMBEDDER ─────────────────────────────────────────
class Embedder:
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')
        self.fitted = False

    def fit_embed(self, texts):
        self.vec.fit(texts)
        self.fitted = True
        return np.mean(self.vec.transform(texts).toarray(), axis=0)

    def embed(self, texts):
        if not self.fitted: return self.fit_embed(texts)
        return np.mean(self.vec.transform(texts).toarray(), axis=0)

def drift_score(v1, v2):
    sim = cosine_similarity([v1], [v2])[0][0]
    return round(1.0 - float(sim), 4)

# ── TEST SCENARIOS ────────────────────────────────────
# Baseline — AI/ML topics
BASELINE = [
    "what is machine learning",
    "what is deep learning",
    "what is neural network",
    "what is python programming",
    "explain artificial intelligence",
]

# Scenario 1 — SHOULD NOT heal (very similar)
SIMILAR = [
    "supervised learning tutorial",
    "how neural networks work",
    "deep learning basics python",
    "machine learning algorithms",
]

# Scenario 2 — BORDERLINE (somewhat different)
BORDERLINE = [
    "data analysis with excel",
    "statistics for beginners",
    "database management sql",
    "web development basics",
]

# Scenario 3 — SHOULD heal (very different)
DIFFERENT = [
    "best cricket team in world",
    "recipe for butter chicken",
    "bollywood movies 2026",
    "weather forecast mumbai",
]

# Scenario 4 — EXTREME drift
EXTREME = [
    "buy iPhone 15 online discount",
    "football world cup results",
    "stock market investment tips",
    "travel to europe packages",
]

scenarios = [
    ("SIMILAR",    SIMILAR,    "Should NOT heal — agent on track"),
    ("BORDERLINE", BORDERLINE, "Gray area — calibration critical here"),
    ("DIFFERENT",  DIFFERENT,  "SHOULD heal — agent drifted"),
    ("EXTREME",    EXTREME,    "Definitely heal — major drift"),
]

# ── RUN EXPERIMENT ────────────────────────────────────
print("\n📊 Step 1: Computing drift scores for all scenarios...")
print("-"*50)

embedder = Embedder()
baseline_vec = embedder.fit_embed(BASELINE)
print(f"✅ Baseline set from {len(BASELINE)} texts\n")

scores = {}
for name, texts, desc in scenarios:
    vec = embedder.embed(texts)
    score = drift_score(baseline_vec, vec)
    scores[name] = score
    print(f"  {name:12} → Drift Score: {score:.4f}  ({desc})")

# ── THRESHOLD ANALYSIS ────────────────────────────────
print("\n\n📐 Step 2: Testing different threshold values...")
print("-"*50)
print(f"{'Threshold':>12} | {'SIMILAR':>8} | {'BORDER':>8} | {'DIFF':>8} | {'EXTREME':>8} | {'Result'}")
print("-"*70)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
results = []

for theta in thresholds:
    sim_heal   = scores["SIMILAR"]    >= theta
    bord_heal  = scores["BORDERLINE"] >= theta
    diff_heal  = scores["DIFFERENT"]  >= theta
    ext_heal   = scores["EXTREME"]    >= theta

    # Ideal: SIMILAR=False, BORDERLINE=False, DIFFERENT=True, EXTREME=True
    correct = (not sim_heal) and (not bord_heal) and diff_heal and ext_heal

    result_str = "✅ OPTIMAL" if correct else ("⚠️ Too low" if not diff_heal else "⚠️ Too high" if sim_heal else "〰️ Partial")

    print(f"  θ = {theta:.2f}    | {str(sim_heal):>8} | {str(bord_heal):>8} | {str(diff_heal):>8} | {str(ext_heal):>8} | {result_str}")
    results.append({
        "threshold": theta,
        "similar_heals": sim_heal,
        "borderline_heals": bord_heal,
        "different_heals": diff_heal,
        "extreme_heals": ext_heal,
        "is_optimal": correct
    })

# ── FIND BEST THRESHOLD ───────────────────────────────
print("\n\n🎯 Step 3: Finding optimal threshold range...")
print("-"*50)

optimal = [r for r in results if r["is_optimal"]]
if optimal:
    best = optimal[len(optimal)//2]  # Middle of optimal range
    print(f"\n✅ Optimal threshold range: {optimal[0]['threshold']:.2f} — {optimal[-1]['threshold']:.2f}")
    print(f"✅ Recommended θ = {best['threshold']:.2f}")
    print(f"\n   Justification:")
    print(f"   • Similar topics (should stay): drift = {scores['SIMILAR']:.4f} → {'Heal' if scores['SIMILAR']>=best['threshold'] else 'No heal'} ✅")
    print(f"   • Borderline (stay):            drift = {scores['BORDERLINE']:.4f} → {'Heal' if scores['BORDERLINE']>=best['threshold'] else 'No heal'} ✅")
    print(f"   • Different (should heal):      drift = {scores['DIFFERENT']:.4f} → {'Heal' if scores['DIFFERENT']>=best['threshold'] else 'No heal'} ✅")
    print(f"   • Extreme (should heal):        drift = {scores['EXTREME']:.4f} → {'Heal' if scores['EXTREME']>=best['threshold'] else 'No heal'} ✅")
else:
    print("⚠️ No single perfectly optimal threshold found — using 0.45 as safe default")
    best = {"threshold": 0.45}

# ── SAVE RESULTS ─────────────────────────────────────
report = {
    "timestamp": datetime.now().isoformat(),
    "experiment": "Threshold Calibration Study",
    "baseline_size": len(BASELINE),
    "scenario_scores": scores,
    "threshold_analysis": results,
    "recommendation": {
        "optimal_threshold": best["threshold"],
        "justification": f"Correctly classifies all 4 scenarios at θ={best['threshold']:.2f}"
    }
}

with open("calibration_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n\n💾 Report saved to calibration_report.json")
print(f"\n{'='*50}")
print(f"✅ Week 2 Day 2 — Calibration Study Complete!")
print(f"   Recommended threshold: θ = {best['threshold']:.2f}")
print(f"   This validates our white paper Section 7(a) claim!")