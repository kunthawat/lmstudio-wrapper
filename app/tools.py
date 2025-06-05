def search(query: str) -> str:
    return f"Search result for '{query}'"

search_tool = Tool(
    name="Search",
    func=search,
    description="Useful for searching information."
)