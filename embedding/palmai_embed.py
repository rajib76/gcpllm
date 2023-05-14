import os
from typing import List

from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from vertexai.language_models._language_models import TextEmbeddingModel

from embedding.llm_embed import LLMEmbed

load_dotenv()
gcp_apl_cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
gcp_project = os.getenv("GCP_PROJECT")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_apl_cred
os.environ["GCP_PROJECT"] = gcp_project

class Embedding(LLMEmbed):
    def __init__(self, model_id):
        super().__init__()
        self.module = "Palm AI"
        palmai = TextEmbeddingModel.from_pretrained(model_id)
        self.embed_model = palmai

    def embed_text_pairs(self, text1="", text2="") -> List:
        print("Embedding with {model}:".format(model=self.module))
        embeddigs = self.embed_model.get_embeddings([text1])
        for embedding in embeddigs:
            emb_text_1 = embedding.values

        embeddigs = self.embed_model.get_embeddings([text2])
        for embedding in embeddigs:
            emb_text_2 = embedding.values

        return emb_text_1,emb_text_2

    def get_score(self, text1, text2):
        emb_text_1, emb_text_2 = self.embed_text_pairs(text1, text2)
        return self.get_similarity_score(emb_text_1, emb_text_2)


if __name__ == "__main__":
    cemb = Embedding("textembedding-gecko@001")
    cemb.get_similarity_score()
