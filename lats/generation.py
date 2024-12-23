from typing import Optional, Dict, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from .reflection_utils import Reflection


class ExpansionResult(BaseModel):
    """Result from expanding a node."""

    success: bool = Field(default=False)
    reflection: Optional["Reflection"] = None
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.success and self.reflection is not None


# Create expansion prompt
expansion_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helping to solve the query: {query}"),
        ("human", "{messages[-1].content}"),
    ]
)

# Create expansion chain
expansion_llm = ChatOpenAI(temperature=0.7)


async def expansion_chain(inputs: Dict[str, Any]) -> ExpansionResult:
    try:
        result = await (
            RunnablePassthrough() | expansion_prompt | expansion_llm
        ).ainvoke(inputs)

        return ExpansionResult(
            success=True,
            reflection=result.reflection,
        )
    except Exception as e:
        return ExpansionResult(success=False, error=str(e))
