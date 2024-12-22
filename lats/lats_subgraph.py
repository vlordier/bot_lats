from langchain.chains import LLMChain


class LATSSubgraph:
    def __init__(self, llm_chain: LLMChain):
        self.llm_chain = llm_chain

    def perform_task(self, input_data: str) -> str:
        return self.llm_chain.invoke(f"LATS analysis: {input_data}")
