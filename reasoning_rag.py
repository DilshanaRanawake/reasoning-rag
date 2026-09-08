# reasoning_rag.py
from typing import TypedDict, List
import json
from openai import OpenAI

from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
LLM_MODEL = "llama-3.2-3b-instruct"
model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.PersistentClient(path="./chroma_store")
collection = client_db.get_collection("research_papers")

MAX_ITERATIONS = 3

class RAGState(TypedDict):
    question: str
    search_history: List[str]
    retrieved: List[str]
    next_action: str
    next_query: str
    answer: str
    iterations: int

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def decide_node(state: RAGState) -> dict:
    context = "\n\n".join(state["retrieved"]) or "(nothing retrieved yet)"
    prompt = f"""Question: {state['question']}
Already retrieved:
{context}

Previous search queries tried: {state['search_history']}

Decide: do you have enough information to answer confidently, or do you need to search
the research papers again? Respond ONLY as JSON, no markdown fences, no extra text:
{{"action": "search or answer", "query": "search query if action is search else empty"}}"""

    raw = call_llm(prompt).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {"action": "answer", "query": ""}
    return {"next_action": decision.get("action", "answer"), "next_query": decision.get("query", "")}

def retrieve_node(state: RAGState) -> dict:
    query = state["next_query"]
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    new_chunks = results["documents"][0]
    return {
        "retrieved": state["retrieved"] + new_chunks,
        "search_history": state["search_history"] + [query],
        "iterations": state["iterations"] + 1,
    }

def answer_node(state: RAGState) -> dict:
    context = "\n\n".join(state["retrieved"]) or "(no context retrieved)"
    prompt = f"""Answer this question using only the context below. If the context is
insufficient, say so honestly.

Context:
{context}

Question: {state['question']}"""
    return {"answer": call_llm(prompt)}

def route(state: RAGState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "answer"
    return state.get("next_action", "answer")

graph = StateGraph(RAGState)
graph.add_node("decide", decide_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("decide")
graph.add_conditional_edges("decide", route, {"search": "retrieve", "answer": "answer"})
graph.add_edge("retrieve", "decide")
graph.add_edge("answer", END)

app = graph.compile()

if __name__ == "__main__":
    question = input("Ask a question about your papers: ")
    result = app.invoke({
        "question": question,
        "search_history": [],
        "retrieved": [],
        "next_action": "",
        "next_query": "",
        "answer": "",
        "iterations": 0,
    })
    print("\n--- Search queries used ---")
    print(result["search_history"])
    print("\n--- Answer ---")
    print(result["answer"])