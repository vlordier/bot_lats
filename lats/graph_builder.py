from langgraph.graph import END, StateGraph, START
from typing import Any, Dict
from .tree import SearchState
from .tools import generate_initial_response


# Move these functions here to avoid circular imports
async def expand_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the current node."""
    root = state["root"]
    # Implement basic expansion
    return {"root": root, "query": state["query"], "expanded": True}


def should_continue_search(state: SearchState) -> str:
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"


builder = StateGraph(SearchState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand_node)
builder.add_edge(START, "start")

builder.add_conditional_edges(
    "start",
    should_continue_search,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    should_continue_search,
    ["expand", END],
)

search_graph = builder.compile()
