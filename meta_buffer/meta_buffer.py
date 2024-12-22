import os
from typing import List, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain.chains import LLMChain
from langchain_community.docstore.in_memory import InMemoryDocstore
from reasoner.thought_template import ThoughtTemplate


class MetaBuffer:
    def __init__(self, llm_chain: LLMChain, embedding_model: OpenAIEmbeddings):
        self.llm_chain = llm_chain
        self.embedding_model = embedding_model
        self.templates: List[ThoughtTemplate] = []
        if os.path.exists("./meta_buffer_faiss"):
            self.vector_store = FAISS.load_local("./meta_buffer_faiss", embedding_model)
        else:
            # Determine the dimension using a sample embedding
            sample_embedding = embedding_model.embed_query("sample text")
            index = faiss.IndexFlatL2(len(sample_embedding))
            self.vector_store = FAISS(
                embedding_function=embedding_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    def retrieve_and_instantiate(self, input_text: str) -> Optional[ThoughtTemplate]:
        input_embedding = self.embedding_model.embed_query(input_text)
        results = self.vector_store.similarity_search_by_vector(input_embedding, k=5)
        if results:
            best_match = max(results, key=lambda res: res.score)
            return best_match.metadata["template"]
        return None

    def dynamic_update(self, thought_template: ThoughtTemplate):
        input_embedding = self.embedding_model.embed_query(str(thought_template.dict()))
        results = self.vector_store.similarity_search_by_vector(input_embedding, k=5)
        if results and any(res.score > 0.85 for res in results):
            self._merge_templates(results[0].metadata["template"], thought_template)
        else:
            self.templates.append(thought_template)
            self.vector_store.add_texts(
                [str(thought_template.dict())],
                metadatas=[{"template": thought_template}],
            )

    def _merge_templates(self, existing: ThoughtTemplate, new: ThoughtTemplate):
        existing.nodes = list(set(existing.nodes + new.nodes))
        existing.edges = list(set(existing.edges + new.edges))
        existing.labels.update(new.labels)
