# version.py — MemoryOS Final Version Info

PROJECT = {
    "name":        "MemoryOS for AI Agents",
    "version":     "1.0.0",
    "description": "Self-Healing Memory Architecture with Real-Time Drift Detection",
    "authors":     ["Aaditya Rana", "Shivam Kumar Poddar"],
    "university":  "Parul University",
    "department":  "CSE-AI",
    "supervisor":  "Ms. Kiran Sharma",
    "github":      "github.com/shivampoddar12/memoryos-project",
    "semester":    "6th Semester Major Project",
    "year":        "2025-26",
    "key_results": {
        "drift_reduction":    "90.9%",
        "consistency_gain":   "64.7%",
        "modules_built":      14,
        "weeks_completed":    3,
        "papers_cited":       6,
    }
}

if __name__ == "__main__":
    print("="*50)
    print(f"  {PROJECT['name']} v{PROJECT['version']}")
    print("="*50)
    for k, v in PROJECT.items():
        if k != "key_results":
            print(f"  {k:15}: {v}")
    print("\n  Key Results:")
    for k, v in PROJECT["key_results"].items():
        print(f"    {k:22}: {v}")
    print("="*50)