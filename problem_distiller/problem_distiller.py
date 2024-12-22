from langchain.chains import LLMChain
from pydantic import BaseModel


class ProblemInfo(BaseModel):
    nodes: list[str]
    edges: list[tuple]
    labels: dict[str, str]
    distilled_info: dict


class ProblemDistiller:
    def __init__(self, llm_chain: LLMChain):
        self.llm_chain = llm_chain

    def distill(self, problem: str) -> ProblemInfo:
        response = self.llm_chain.invoke(f"Distill problem: {problem}")
        distilled_info = (
            response if isinstance(response, dict) else {"text": str(response)}
        )
        return ProblemInfo(
            nodes=["input_1", "input_2", "output"],
            edges=[("input_1", "output"), ("input_2", "output")],
            labels={"output": "function(input_1, input_2)"},
            distilled_info=distilled_info,
        )
