from langgraph.graph import END, StateGraph, START
from .tree import SearchState
from .search_agent import LATSAgent  # Updated import
from .tools import generate_initial_response

builder = StateGraph(SearchState)  # Consistent state type
builder.add_node("start", generate_initial_response)
builder.add_node("expand", LATSAgent.expand_node)  # Updated method name
builder.add_edge(START, "start")

builder.add_conditional_edges(
    "start",
    LATSAgent.should_continue_search,  # Fixed class name
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    LATSAgent.should_continue_search,  # Fixed class name
    ["expand", END],
)

search_graph = builder.compile()  # Renamed from graph
