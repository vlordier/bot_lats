from typing import List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.runnables import as_runnable


class ReflectionMetrics(BaseModel):
    token_count: int = Field(ge=0)
    response_time: float = Field(ge=0.0)
    confidence_scores: List[float] = Field(default_factory=list)

    @field_validator("confidence_scores")
    @classmethod
    def validate_scores(cls, v: List[float]) -> List[float]:
        return [round(score, 3) for score in v]


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )
    coherence_score: float = Field(
        description="Score for logical flow and coherence", ge=0, le=1
    )
    relevance_score: float = Field(
        description="Score for relevance to query", ge=0, le=1
    )
    novelty_score: float = Field(
        description="Score for introducing new useful info", ge=0, le=1
    )
    metrics: ReflectionMetrics = Field(default_factory=ReflectionMetrics)

    @field_validator("coherence_score", "relevance_score", "novelty_score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Score must be between 0 and 1")
        return round(v, 3)  # Normalize to 3 decimal places

    @model_validator(mode="after")
    def validate_found_solution_score(self) -> "Reflection":
        if self.found_solution and self.score < 7:
            raise ValueError("Solution found but score too low")
        return self

    @model_validator(mode="after")
    def validate_metrics(self) -> "Reflection":
        if self.found_solution and not self.metrics.confidence_scores:
            raise ValueError("Solution found but no confidence scores")
        return self

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        weights = {"base": 0.4, "coherence": 0.2, "relevance": 0.3, "novelty": 0.1}
        return (
            weights["base"] * (self.score / 10.0)
            + weights["coherence"] * self.coherence_score
            + weights["relevance"] * self.relevance_score
            + weights["novelty"] * self.novelty_score
        )


# Create reflection LLM chain
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Analyze the following response: {candidate}"),
    ]
)

reflection_llm = ChatOpenAI(temperature=0.7)
reflection_llm_chain = LLMChain(llm=reflection_llm, prompt=reflection_prompt)


@as_runnable
def reflection_chain(inputs) -> Reflection:
    tool_choices = reflection_llm_chain.invoke(inputs)
    reflection = tool_choices[0]
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection
