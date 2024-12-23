from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict
import math
import asyncio
import logging
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, model_validator, ConfigDict

from contextlib import asynccontextmanager
from .reflection_utils import Reflection


class NodeStats(BaseModel):
    """Statistics for MCTS node."""

    cumulative_value: float = Field(default=0.0)
    visit_count: int = Field(default=0, ge=0)
    depth: int = Field(default=0, ge=0)

    @property
    def average_value(self) -> float:
        return self.cumulative_value / max(1, self.visit_count)


class SimulationConfig(BaseModel):
    num_simulations: int = Field(default=5, gt=0)
    max_depth: int = Field(default=5, gt=0)
    decay_factor: float = Field(default=0.95, gt=0, lt=1)

    model_config = ConfigDict(frozen=True)


class SimulationResult(BaseModel):
    value: float
    depth_reached: int = Field(ge=0)
    terminated_early: bool = False

    @model_validator(mode="after")
    def validate_result(self) -> "SimulationResult":
        if self.terminated_early and self.depth_reached == 0:
            raise ValueError("Early termination requires non-zero depth")
        return self


class MCTSNode(BaseModel):
    """Pydantic model for MCTS nodes."""

    messages: List[BaseMessage]
    reflection: Reflection
    stats: NodeStats = Field(default_factory=NodeStats)
    parent: Optional["MCTSNode"]
    children: List["MCTSNode"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tree_structure(self) -> "MCTSNode":
        for child in self.children:
            if child.parent is not self:
                raise ValueError("Invalid tree structure: parent-child mismatch")
        return self

    @property
    def parent_stats(self) -> Optional[NodeStats]:
        """Safely access parent stats."""
        return self.parent.stats if self.parent else None

    # Methods can remain largely the same, just updated to use stats
    def update(self, value: float) -> None:
        self.stats.cumulative_value += value
        self.stats.visit_count += 1

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        """Calculate UCB1 value for node selection."""
        if self.stats.visit_count == 0:
            return float("inf")

        parent_visits = self.parent_stats.visit_count if self.parent_stats else 1
        exploitation = self.stats.cumulative_value / self.stats.visit_count
        exploration = math.sqrt(math.log(parent_visits) / self.stats.visit_count)
        return exploitation + exploration_weight * exploration

    @property
    def value(self) -> float:
        """Get node's average value."""
        return self.stats.cumulative_value / max(1, self.stats.visit_count)

    @property
    def is_solved(self) -> bool:
        """Check if node represents a complete solution."""
        return self.reflection.found_solution

    @property
    def height(self) -> int:
        """Get tree height from this node."""
        if not self.children:
            return 0
        return 1 + max(child.height for child in self.children)

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        """Get message history from root to this node."""
        messages = []
        current = self
        while current:
            messages.extend(current.messages)
            if include_reflections:
                messages.append(current.reflection.as_message())
            current = current.parent
        return list(reversed(messages))

    @staticmethod
    def select(root: "MCTSNode") -> "MCTSNode":
        """Select most promising node for expansion."""
        current = root
        while current.children:
            try:
                children = [c for c in current.children if c is not None]
                if not children:
                    break
                best_child = max(children, key=lambda c: c.upper_confidence_bound())
                current = best_child
            except ValueError:
                break
        return current

    def get_best_solution(self) -> "MCTSNode":
        """Get best solution node in subtree."""
        if not self.children:
            return self
        return max(
            (child.get_best_solution() for child in self.children),
            key=lambda n: n.value,
        )

    async def simulate_parallel(
        self, state: Dict[str, Any], num_simulations: int = 5
    ) -> List[float]:
        """Run multiple simulations in parallel."""
        config = SimulationConfig(num_simulations=num_simulations)
        tasks = [self._simulate(state, config) for _ in range(num_simulations)]
        return await asyncio.gather(*tasks)

    async def get_candidate_nodes(
        self, query: str, num_candidates: int = 3
    ) -> List["MCTSNode"]:
        """Generate candidate child nodes."""
        if self.is_solved:
            return []

        candidates = []
        for _ in range(num_candidates):
            child = MCTSNode(
                messages=self.messages.copy(),
                reflection=Reflection(
                    reflections="Candidate expansion",
                    score=0,
                    found_solution=False,
                    coherence_score=0.0,
                    relevance_score=0.0,
                    novelty_score=0.0,
                ),
                parent=self,
            )
            candidates.append(child)
            self.children.append(child)

        return candidates

    async def _simulate(
        self, state: Dict[str, Any], config: SimulationConfig
    ) -> SimulationResult:
        """Enhanced simulation with actual response generation."""
        async with self._simulation_context() as ctx:
            try:
                timeout = getattr(config, "timeout", 30.0)
                result = await asyncio.wait_for(
                    self._run_simulation(state, config, ctx), timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                return SimulationResult(
                    value=self.value * config.decay_factor,
                    depth_reached=1,
                    terminated_early=True,
                )
            except Exception as e:
                logging.error(f"Simulation failed: {str(e)}")
                return SimulationResult(
                    value=0.0, depth_reached=0, terminated_early=True
                )

    @asynccontextmanager
    async def _simulation_context(self):
        """Resource management for simulations."""
        try:
            yield {"start_time": asyncio.get_event_loop().time()}
        finally:
            # Cleanup resources if needed
            pass

    def calculate_reward(self, reflection: Reflection) -> float:
        """Calculate composite reward from reflection."""
        solution_bonus = 2.0 if reflection.found_solution else 0.0
        quality_score = reflection.normalized_score
        depth_penalty = 0.9**self.height

        return (quality_score + solution_bonus) * depth_penalty

    def backpropagate(self, value: float) -> None:
        """Backpropagate value through ancestors."""
        node = self
        while node:
            node.update(value)
            node = node.parent

    def select_best_child(self) -> Optional["MCTSNode"]:
        """Select child with highest value for actual moves."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.value)

    @staticmethod
    async def select_and_expand(root: "MCTSNode", state: Dict[str, Any]) -> "MCTSNode":
        """Combined selection and expansion with parallel simulation."""
        selected = MCTSNode.select(root)
        values = await selected.simulate_parallel(state)
        for value in values:
            selected.backpropagate(value)

        best_child = selected.select_best_child()
        if best_child is None:
            return selected
        return best_child


class SearchState(TypedDict):
    root: MCTSNode
    query: str
