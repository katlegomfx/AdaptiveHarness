# src/agent/context.py
from src.llm_backend import get_embedding
from src.memory.storage import retrieve_learnings, retrieve_aspect_memory


def build_aspect_context(agent, aspect: str, query: str, recent_n=4):
    """
    Builds a tailored context list for a specific sub-agent (aspect).
    Retrieves: Recent chat + Global learnings + Aspect's own past memories.
    """
    context = []

    # 1. Get recent chat history (so it knows what's happening right now)
    start_idx = 1 if agent.history and agent.history[0].get(
        "role") == "system" else 0
    context.extend(agent.history[start_idx:][-recent_n:])

    # 2. Get aspect-specific past memories (e.g., past plans, past test cases)
    query_emb = get_embedding(query)
    aspect_mems = retrieve_aspect_memory(
        aspect, agent.session_id, query_emb=query_emb, limit=2)
    if aspect_mems:
        mem_block = "\n".join(f"- {m}" for m in aspect_mems)
        context.append(
            {"role": "user", "content": f"Past {aspect} memories for similar tasks:\n{mem_block}"})

    # 3. Get relevant global learnings
    learnings = retrieve_learnings(query, query_emb=query_emb, limit=2)
    if learnings:
        learn_block = "\n".join(f"- {l}" for l in learnings)
        context.append(
            {"role": "user", "content": f"Relevant Past Learnings:\n{learn_block}"})

    return context
