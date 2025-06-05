# app/tools.py
from langchain.tools import Tool  # ← Add this line

def search(query: str) -> str:
    return f"Search result for '{query}'"

search_tool = Tool(
    name="Search",
    func=search,
    description="Useful for searching information."
)