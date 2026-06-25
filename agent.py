# agent.py — MemoryOS Week 1, Day 1
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json
from datetime import datetime

# Agent ki memory
class AgentState(TypedDict):
    messages: List[str]
    memory_log: List[dict]
    session_count: int

# Simple response (bina OpenAI key ke bhi kaam karega)
def agent_node(state: AgentState) -> AgentState:
    user_msg = state["messages"][-1]
    
    # Simple echo response (Day 1 ke liye)
    response = f"Agent received: '{user_msg}' | Memory entries: {len(state['memory_log'])}"
    
    entry = {
        "session": state["session_count"],
        "time": datetime.now().isoformat(),
        "input": user_msg,
        "response": response
    }
    
    print(f"\n🧠 Memory Entry #{state['session_count']}")
    print(f"   Input: {user_msg}")
    print(f"   Stored memories: {len(state['memory_log']) + 1}")
    
    return {
        "messages": state["messages"],
        "memory_log": state["memory_log"] + [entry],
        "session_count": state["session_count"] + 1
    }

# Graph banao
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)
agent = graph.compile()

# Run karo
if __name__ == "__main__":
    state = {"messages": [], "memory_log": [], "session_count": 1}
    
    print("🤖 MemoryOS Agent — Day 1")
    print("="*35)
    print("'quit' likho band karne ke liye\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            with open("memory_log.json", "w") as f:
                json.dump(state["memory_log"], f, indent=2)
            print(f"\n💾 Memory saved! ({len(state['memory_log'])} entries)")
            break
        
        state["messages"].append(user_input)
        state = agent.invoke(state)
        print(f"🤖 {state['memory_log'][-1]['response']}\n")