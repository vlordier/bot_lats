from langchain.embeddings import OpenAIEmbeddings


class EmbeddingUtils:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings()

    def embed_text(self, text: str):
        return self.embedding_model.embed_text(text)
