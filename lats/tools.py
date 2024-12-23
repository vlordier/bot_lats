from typing import List, Tuple, Optional, Any, Dict
import logging
from langchain.errors import LangChainError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import SearchConfig
from .tree import MCTSNode
from .reflection_utils import Reflection


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _execute_tool_with_retry(tool: Any, **kwargs) -> Any:
    """Execute tool with retry logic."""
    try:
        return await tool.arun(**kwargs)
    except Exception as e:
        logging.error(f"Tool execution failed: {str(e)}")
        raise


def setup_tools(
    config: SearchConfig,
) -> Tuple[Optional[ChatOpenAI], List[TavilySearchResults], Optional[ToolNode]]:
    """Initialize LLM and search tools with error handling."""
    try:
        llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_timeout=config.timeout,
        )

        if config.tavily_api_key:
            search = TavilySearchAPIWrapper(
                tavily_api_key=config.tavily_api_key,
                max_results=config.branching_factor,
            )
        else:
            search = TavilySearchAPIWrapper()

        tavily_tool = TavilySearchResults(api_wrapper=search)
        tool_node = ToolNode(tools=[tavily_tool])

        return llm, [tavily_tool], tool_node

    except LangChainError as e:
        logging.error(f"Error setting up tools: {e}")
        return None, [], None


async def generate_initial_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate initial response for search query."""
    try:
        root = MCTSNode(
            messages=[HumanMessage(content=state["query"])],
            reflection=Reflection(
                reflections="Initial query",
                score=5,
                found_solution=False,
                coherence_score=1.0,
                relevance_score=1.0,
                novelty_score=0.5,
            ),
        )
        return {"root": root, "query": state["query"]}
    except Exception as e:
        logging.error(f"Error generating initial response: {e}")
        raise
