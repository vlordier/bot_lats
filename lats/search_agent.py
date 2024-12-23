from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langchain.errors import LangChainError
import asyncio
from functools import lru_cache
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .config import LATSConfig
from .graph_builder import search_graph
from .tree import SearchState
from .generation import expansion_chain


class SearchMetadata(BaseModel):
    total_nodes: int = Field(default=0, ge=0)
    max_depth_reached: int = Field(default=0, ge=0)
    computation_time: float = Field(default=0.0, ge=0.0)
    model_config = ConfigDict(frozen=True)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_depth: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timeout: Optional[float] = Field(default=None, gt=0)

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("Query too short")
        return v.strip()


class SearchResponse(BaseModel):
    solution: str
    confidence: float = Field(ge=0.0, le=1.0)
    num_steps: int = Field(ge=0)
    is_complete: bool
    full_trajectory: Optional[List[BaseMessage]] = None
    error: Optional[str] = None
    metadata: SearchMetadata = Field(default_factory=SearchMetadata)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(v, 3)


class LATSAgent:
    """Monte Carlo Tree Search Agent for Language Tasks"""

    def __init__(self, config: Optional[LATSConfig] = None):
        self.config = config or LATSConfig()
        self.graph = search_graph
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=1000)
    def _cached_search(self, request: SearchRequest) -> Dict[str, Any]:
        """Cached version of search results."""
        return self.search(request)

    async def async_search(self, request: SearchRequest) -> Dict[str, Any]:
        """Asynchronous version of search method."""
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.search, request), timeout=self.config.timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error("Search timed out")
            return {"error": "Search timed out"}
        except Exception as e:
            self.logger.error(f"Async search failed: {str(e)}")
            return {"error": f"Async search failed: {str(e)}"}

    def batch_search(self, requests: List[SearchRequest]) -> List[Dict[str, Any]]:
        """Execute multiple searches in parallel."""

        async def _batch():
            tasks = [self.async_search(r) for r in requests]
            return await asyncio.gather(*tasks)

        return asyncio.run(_batch())

    def _get_temperature(self, depth: int) -> float:
        """Dynamic temperature scheduling based on search depth."""
        return self.config.temperature * (0.8**depth)

    @staticmethod
    async def expand_node(state: SearchState, config: RunnableConfig) -> SearchState:
        """Expand nodes with dynamic temperature and parallel processing."""
        root = state["root"]
        depth = root.height

        candidates = await root.get_candidate_nodes(state["query"])
        temp = 0.7 * (0.8**depth)
        config_dict = {**config, "temperature": temp}

        batch_size = min(getattr(config, "max_concurrent", 5), len(candidates))
        batches = [
            candidates[i : i + batch_size]
            for i in range(0, len(candidates), batch_size)
        ]

        child_nodes = []
        evaluations = []

        for batch in batches:
            tasks = [
                expansion_chain(
                    {
                        "messages": node.messages,
                        "query": state["query"],
                        "config": config_dict,
                    }
                )
                for node in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for node, result in zip(batch, batch_results):
                if result.success:
                    child_nodes.append(node)
                    evaluations.append(result.reflection)

        for node, evaluation in zip(child_nodes, evaluations):
            node.backpropagate(evaluation.normalized_score)

        return state

    @staticmethod
    def should_continue_search(state: SearchState) -> str:
        """Determine whether to continue tree search."""
        root = state["root"]
        if root.is_solved:
            return END
        if root.height > 5:
            return END
        return "expand"

    def search(self, request: SearchRequest) -> SearchResponse:
        """Run search on input query."""
        try:
            last_step = None
            for step in self.graph.stream({"query": request.query}):
                if step is None:
                    break
                last_step = step

            if last_step is None:
                return SearchResponse(error="No solution found")

            solution_node = last_step.get("expand", {}).get("root")
            if solution_node is None:
                return SearchResponse(error="No valid solution node")

            best_solution = solution_node.get_best_solution()
            trajectory = best_solution.get_trajectory(include_reflections=False)

            return SearchResponse(
                solution=trajectory[-1].content,
                confidence=best_solution.value,
                num_steps=best_solution.depth,
                is_complete=best_solution.is_solved,
                full_trajectory=trajectory,
            )

        except LangChainError as e:
            return SearchResponse(error=f"LangChain error: {str(e)}")
        except Exception as e:
            return SearchResponse(error=f"Unexpected error: {str(e)}")
