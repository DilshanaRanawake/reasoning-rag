# server.py
import os

# Must be set before importing sentence_transformers / huggingface_hub.
# Without this, model loading tries to reach huggingface.co on startup;
# in Claude Desktop's own process (no inherited terminal env vars) that
# network check can hang or stall long enough that the MCP client's
# `initialize` request times out and the server is marked "Failed".
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server.mcpserver import MCPServer

# Anchor all file paths to this script's own location, so it works no matter
# what directory the process is launched from (terminal vs Claude Desktop).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_store")

# Create server (mcp v2.x: FastMCP was renamed to MCPServer)
server = MCPServer("research-papers")

# Lazy-load the embedding model and Chroma collection on first use, NOT at
# import time. Claude Desktop's MCP client expects an `initialize` response
# within ~60s; importing torch/sentence-transformers and loading the model
# eagerly here can blow past that on a cold start (Defender scanning, slow
# disk, first-time model load), leaving the server stuck "Failed" even
# though it would have worked fine given more time. Loading on first call
# lets `initialize` return instantly; only the first actual tool call pays
# the startup cost.
_model = None
_collection = None


def _get_model_and_collection():
    global _model, _collection
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection("research_papers")
    return _model, _collection


@server.tool()
def search_papers(query: str, top_k: int = 3) -> str:
    """Search across all papers and return the most relevant passages,
    tagged with which paper each one came from."""
    model, collection = _get_model_and_collection()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    output = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(f"[{meta['paper']}]\n{doc}")
    return "\n\n---\n\n".join(output) if output else "No relevant passages found."


@server.tool()
def list_papers() -> str:
    """List all papers available in this research server."""
    _, collection = _get_model_and_collection()
    all_meta = collection.get()["metadatas"]
    papers = sorted(set(m["paper"] for m in all_meta))
    return "\n".join(papers)


@server.tool()
def get_paper_chunks(paper_id: str) -> str:
    """Get all text chunks belonging to one specific paper, by its id
    (use list_papers to see valid ids)."""
    _, collection = _get_model_and_collection()
    results = collection.get(where={"paper": paper_id})
    if not results["documents"]:
        return f"No paper found with id '{paper_id}'."
    return "\n\n".join(results["documents"])


if __name__ == "__main__":
    server.run()