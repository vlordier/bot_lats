from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
import logging


class SearchConfig(BaseModel):
    """Configuration for search functionality."""

    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    tavily_api_key: Optional[str] = None
    branching_factor: int = Field(default=3, gt=0)


class LATSConfig(BaseModel):
    """Configuration for Language Agent Tree Search (LATS)."""

    # MCTS parameters
    max_search_depth: int = Field(default=5, gt=0, le=10)
    branching_factor: int = Field(default=5, gt=0, le=10)
    exploration_weight: float = Field(default=1.0, gt=0.0, le=2.0)
    max_iterations: int = Field(default=100, gt=0)

    # Model settings
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)

    # API keys
    tavily_api_key: Optional[str] = None

    # Additional settings
    max_tokens: int = 2000
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_responses: bool = True

    # Caching settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Performance settings
    batch_size: int = 10
    max_concurrent: int = 5

    # Custom model configs
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # MCTS simulation parameters
    simulation_depth: int = 5
    discount_factor: float = 0.95
    exploration_chance: float = 0.1

    # Temperature scheduling
    min_temperature: float = 0.2
    temperature_decay: float = 0.8

    # Reward calculation
    solution_bonus: float = 2.0
    depth_penalty_factor: float = 0.9

    # Scoring weights
    coherence_weight: float = 0.2
    relevance_weight: float = 0.3
    novelty_weight: float = 0.1
    base_score_weight: float = 0.4

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        allowed_models = {"gpt-4", "gpt-3.5-turbo", "gpt-4-32k"}
        if v not in allowed_models:
            raise ValueError(f"Model must be one of: {allowed_models}")
        return v

    @field_validator("temperature_decay")
    @classmethod
    def validate_decay(cls, v: float, values: Dict[str, Any]) -> float:
        if v >= 1.0:
            raise ValueError("Temperature decay must be less than 1.0")
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "LATSConfig":
        total = (
            self.coherence_weight
            + self.relevance_weight
            + self.novelty_weight
            + self.base_score_weight
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("Scoring weights must sum to 1.0")
        return self

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        logging.basicConfig(
            level=self.log_level,
            filename=self.log_file,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
