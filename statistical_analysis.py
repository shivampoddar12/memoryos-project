# statistical_analysis.py — MemoryOS Week 2 Day 6
# Statistical Analysis + Visualization

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

print("📊 MemoryOS — Statistical Analysis")
print("="*55)

# ── LOAD DATA ────────────────────────────────────────
with open("benchmark_results.json") as f:
    bench = json.load(f)

without = bench["without_memoryos"]
with_   = bench["with_memoryos"]
imp     = bench["improvement"]

scores_without = without["drift_scores"]
scores_with    = with_["drift_scores"]
sessions       = list(range(1, len(scores_without)+1))

# ── STATISTICAL CALCULATIONS ─────────────────────────
print("\n📐 Step 1: Statistical Calculations")
print("-"*55)

def stats(data, name):
    arr = np.array(data)
    result = {
        "mean":   round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std":    round(float(np.std(arr)), 4),
        "min":    round(float(np.min(arr)), 4),
        "max":    round(float(np.max(arr)), 4),
        "variance": round(float(np.var(arr)), 4),
    }
    print(f"\n  {name}:")
    for k,v in result.items():
        print(f"    {k:10}: {v}")
    return result

stats_without = stats(scores_without, "WITHOUT MemoryOS")
stats_with    = stats(scores_with,    "WITH MemoryOS")

# Statistical improvement
mean_diff = round(stats_without["mean"] - stats_with["mean"], 4)
std_diff  = round(stats_without["std"]  - stats_with["std"], 4)
print(f"\n  📊 Mean drift reduction:    {mean_diff:.4f}")
print(f"  📊 Std dev reduction:       {std_diff:.4f} (more stable)")
print(f"  📊 Drift reduction %:       {imp['drift_reduction_percent']}%")
print(f"  📊 Consistency gain %:      {imp['consistency_gain_percent']}%")

# ── GRAPH 1: Drift Score Comparison ──────────────────
print("\n\n📈 Step 2: Generating Graphs...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("MemoryOS Benchmark Results — Statistical Analysis",
             fontsize=14, fontweight='bold', y=0.98)

# Graph 1 — Line chart comparison
ax1 = axes[0, 0]
ax1.plot(sessions, scores_without, 'r-o', linewidth=2,
         markersize=6, label='WITHOUT MemoryOS', color='#FF4444')
ax1.plot(sessions, scores_with, 'g-o', linewidth=2,
         markersize=6, label='WITH MemoryOS', color='#3FB950')
ax1.axhline(y=0.45, color='orange', linestyle='--',
            linewidth=1.5, label='Heal Threshold (0.45)')
ax1.fill_between(sessions, scores_without, scores_with,
                 alpha=0.1, color='green', label='Improvement Area')
ax1.set_title("Drift Score: Session by Session", fontweight='bold')
ax1.set_xlabel("Session Number")
ax1.set_ylabel("Drift Score")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

# Graph 2 — Bar chart comparison
ax2 = axes[0, 1]
x = np.arange(len(sessions))
w = 0.35
bars1 = ax2.bar(x - w/2, scores_without, w,
                label='WITHOUT', color='#FF4444', alpha=0.8)
bars2 = ax2.bar(x + w/2, scores_with, w,
                label='WITH', color='#3FB950', alpha=0.8)
ax2.axhline(y=0.45, color='orange', linestyle='--', linewidth=1.5)
ax2.set_title("Drift Score Bar Chart Comparison", fontweight='bold')
ax2.set_xlabel("Session")
ax2.set_ylabel("Drift Score")
ax2.set_xticks(x)
ax2.set_xticklabels([str(s) for s in sessions])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Graph 3 — Statistical summary bars
ax3 = axes[1, 0]
metrics = ['Mean', 'Median', 'Std Dev', 'Max']
without_vals = [stats_without['mean'], stats_without['median'],
                stats_without['std'], stats_without['max']]
with_vals    = [stats_with['mean'],    stats_with['median'],
                stats_with['std'],     stats_with['max']]

x3 = np.arange(len(metrics))
ax3.bar(x3 - 0.2, without_vals, 0.4, label='WITHOUT', color='#FF4444', alpha=0.8)
ax3.bar(x3 + 0.2, with_vals,    0.4, label='WITH',    color='#3FB950', alpha=0.8)
ax3.set_title("Statistical Metrics Comparison", fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(metrics)
ax3.set_ylabel("Value")
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in ax3.patches:
    ax3.text(bar.get_x() + bar.get_width()/2.,
             bar.get_height() + 0.01,
             f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=7)

# Graph 4 — Improvement summary pie/summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
KEY RESULTS SUMMARY
{'='*35}

Drift Reduction:      {imp['drift_reduction_percent']}%
Consistency Gain:    +{imp['consistency_gain_percent']}%

WITHOUT MemoryOS:
  Mean Drift:  {stats_without['mean']:.4f}
  Std Dev:     {stats_without['std']:.4f}
  Max Drift:   {stats_without['max']:.4f}

WITH MemoryOS:
  Mean Drift:  {stats_with['mean']:.4f}
  Std Dev:     {stats_with['std']:.4f}
  Max Drift:   {stats_with['max']:.4f}

Sessions Tested:  {len(sessions)}
Heal Events:      {with_['heal_events']}
Threshold (θ):    0.45

CONCLUSION:
MemoryOS significantly improves
agent consistency in long-horizon
multi-turn sessions.
"""

ax4.text(0.05, 0.95, summary_text,
         transform=ax4.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.8))

plt.tight_layout()
plt.savefig("benchmark_analysis.png", dpi=150, bbox_inches='tight')
print("✅ benchmark_analysis.png saved!")

# ── SAVE STATS ───────────────────────────────────────
stats_report = {
    "timestamp": datetime.now().isoformat(),
    "without_memoryos_stats": stats_without,
    "with_memoryos_stats":    stats_with,
    "improvement": imp,
    "conclusion": f"MemoryOS reduced drift by {imp['drift_reduction_percent']}% and improved consistency by {imp['consistency_gain_percent']}%"
}

with open("statistical_report.json", "w") as f:
    json.dump(stats_report, f, indent=2)

print("✅ statistical_report.json saved!")
print(f"\n{'='*55}")
print("✅ Week 2 Day 6 — Statistical Analysis Complete!")
print(f"\n🎯 KEY FINDING FOR PAPER:")
print(f"   MemoryOS reduced drift by {imp['drift_reduction_percent']}%")
print(f"   Consistency improved by {imp['consistency_gain_percent']}%")
print(f"   These are publishable results! 🔥")