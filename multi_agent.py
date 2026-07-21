# multi_agent.py — MemoryOS Week 2 Day 3
# Multi-Agent Sync Bus — Shared Knowledge Graph

import numpy as np
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🤖 MemoryOS — Multi-Agent Sync Bus")
print("="*50)

# ── SHARED MEMORY BUS ────────────────────────────────
class SharedMemoryBus:
    """Central knowledge graph — sab agents yahan se share karte hain"""
    
    def __init__(self):
        self.knowledge_graph = {}   # key: topic, value: knowledge
        self.sync_log = []
        self.connected_agents = []
    
    def register_agent(self, agent_id: str):
        self.connected_agents.append(agent_id)
        print(f"   🔗 Agent '{agent_id}' connected to Sync Bus")
    
    def broadcast(self, from_agent: str, key: str, value: str):
        """Ek agent kuch seekhe → sab agents ko mil jaaye"""
        self.knowledge_graph[key] = {
            "value": value,
            "learned_by": from_agent,
            "timestamp": datetime.now().isoformat(),
            "propagated_to": [a for a in self.connected_agents if a != from_agent]
        }
        self.sync_log.append({
            "action": "broadcast",
            "from": from_agent,
            "key": key,
            "timestamp": datetime.now().isoformat()
        })
        print(f"\n   📡 '{from_agent}' broadcast: '{key}'")
        print(f"      Propagated to: {[a for a in self.connected_agents if a != from_agent]}")
    
    def retrieve(self, agent_id: str, key: str) -> str:
        """Agent koi knowledge retrieve kare"""
        if key in self.knowledge_graph:
            data = self.knowledge_graph[key]
            print(f"   ✅ '{agent_id}' retrieved '{key}' (learned by '{data['learned_by']}')")
            return data["value"]
        return None
    
    def get_all_knowledge(self) -> dict:
        return {k: v["value"] for k, v in self.knowledge_graph.items()}
    
    def save(self):
        with open("sync_bus_log.json", "w") as f:
            json.dump({
                "knowledge_graph": self.knowledge_graph,
                "sync_log": self.sync_log,
                "connected_agents": self.connected_agents
            }, f, indent=2)
        print(f"\n   💾 Sync Bus saved ({len(self.knowledge_graph)} knowledge items)")


# ── AGENT CLASS ──────────────────────────────────────
class MemoryAgent:
    def __init__(self, agent_id: str, domain: str, bus: SharedMemoryBus):
        self.agent_id = agent_id
        self.domain = domain
        self.bus = bus
        self.local_memory = []
        self.session_count = 0
        bus.register_agent(agent_id)
        print(f"   🤖 Agent '{agent_id}' initialized | Domain: {domain}")
    
    def learn(self, topic: str, knowledge: str):
        """Agent kuch seekhe aur sab ko broadcast kare"""
        self.local_memory.append({"topic": topic, "knowledge": knowledge})
        self.bus.broadcast(self.agent_id, topic, knowledge)
    
    def ask(self, topic: str) -> str:
        """Pehle local memory check karo, phir shared bus"""
        # Local check
        for mem in self.local_memory:
            if mem["topic"] == topic:
                print(f"   💭 '{self.agent_id}' found '{topic}' in LOCAL memory")
                return mem["knowledge"]
        
        # Shared bus check
        result = self.bus.retrieve(self.agent_id, topic)
        if result:
            # Add to local memory bhi
            self.local_memory.append({"topic": topic, "knowledge": result})
            return result
        
        print(f"   ❓ '{self.agent_id}' doesn't know '{topic}'")
        return None
    
    def get_status(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "local_memories": len(self.local_memory),
            "shared_knowledge": len(self.bus.knowledge_graph)
        }


# ── DEMO ─────────────────────────────────────────────
if __name__ == "__main__":
    
    # Create shared bus
    bus = SharedMemoryBus()
    print("\n📡 Creating 3 specialized agents...")
    print("-"*50)
    
    # 3 specialized agents
    agent_ml   = MemoryAgent("ML-Agent",       "Machine Learning", bus)
    agent_data = MemoryAgent("Data-Agent",     "Data Science",     bus)
    agent_nlp  = MemoryAgent("NLP-Agent",      "NLP & LLMs",       bus)
    
    # ── PHASE 1: Agents learn independently ──────────
    print("\n\n🧠 Phase 1: Agents learning independently...")
    print("-"*50)
    
    agent_ml.learn("gradient_descent",
        "Gradient descent is an optimization algorithm that minimizes loss by iteratively moving in the direction of steepest descent")
    
    agent_data.learn("data_cleaning",
        "Data cleaning involves handling missing values, removing duplicates, fixing data types, and treating outliers")
    
    agent_nlp.learn("transformer_architecture",
        "Transformers use self-attention mechanism with encoder-decoder structure, enabling parallel processing of sequences")
    
    # ── PHASE 2: Knowledge sharing test ──────────────
    print("\n\n🔄 Phase 2: Testing knowledge sharing...")
    print("-"*50)
    
    # ML-Agent puchhe NLP ka knowledge
    print("\n🤔 ML-Agent asking about transformer_architecture...")
    result = agent_ml.ask("transformer_architecture")
    if result:
        print(f"   📚 Got: '{result[:60]}...'")
    
    # NLP-Agent puchhe ML ka knowledge
    print("\n🤔 NLP-Agent asking about gradient_descent...")
    result = agent_nlp.ask("gradient_descent")
    if result:
        print(f"   📚 Got: '{result[:60]}...'")
    
    # Data-Agent puchhe NLP ka knowledge
    print("\n🤔 Data-Agent asking about transformer_architecture...")
    result = agent_data.ask("transformer_architecture")
    if result:
        print(f"   📚 Got: '{result[:60]}...'")
    
    # ── PHASE 3: New knowledge propagation ───────────
    print("\n\n📡 Phase 3: Real-time knowledge propagation...")
    print("-"*50)
    
    print("\n🆕 NLP-Agent learns something new...")
    agent_nlp.learn("context_rot",
        "Context Rot is the degradation of LLM agent performance as session length increases, causing silent production failures")
    
    print("\n🤔 ML-Agent & Data-Agent immediately asking about context_rot...")
    r1 = agent_ml.ask("context_rot")
    r2 = agent_data.ask("context_rot")
    print(f"\n   ML-Agent got it:   {'✅ YES' if r1 else '❌ NO'}")
    print(f"   Data-Agent got it: {'✅ YES' if r2 else '❌ NO'}")
    
    # ── STATUS REPORT ─────────────────────────────────
    print("\n\n📊 Phase 4: Agent Status Report")
    print("-"*50)
    for agent in [agent_ml, agent_data, agent_nlp]:
        status = agent.get_status()
        print(f"\n   🤖 {status['agent_id']:15} | Domain: {status['domain']:20} | "
              f"Local: {status['local_memories']} | Shared: {status['shared_knowledge']}")
    
    print(f"\n   📡 Total shared knowledge items: {len(bus.knowledge_graph)}")
    print(f"   🔗 Connected agents: {bus.connected_agents}")
    
    # ── SAVE ──────────────────────────────────────────
    bus.save()
    
    print(f"\n{'='*50}")
    print("✅ Week 2 Day 3 — Multi-Agent Sync Complete!")
    print("   sync_bus_log.json check karo!")
    print("\n🎯 Key Achievement:")
    print("   → NLP-Agent ne 'context_rot' seekha")
    print("   → ML-Agent + Data-Agent ne turant pa liya")
    print("   → Yahi hai Multi-Agent Shared Memory! 🔥")