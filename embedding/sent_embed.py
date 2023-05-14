import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from embedding.llm_embed import LLMEmbed


class Embedding(LLMEmbed):
    def __init__(self, model_id):
        super().__init__()
        self.module = "Sentence Transformer"
        senttran = SentenceTransformer(model_id)
        self.embed_model = senttran

    def embed_text_pairs(self, text1="", text2=""):
        print("Embedding with {model}:".format(model=self.module))
        emb_text_1 = self.embed_model.encode(text1)
        emb_text_2 = self.embed_model.encode(text2)

        return emb_text_1, emb_text_2

    def get_score(self, text1, text2):
        emb_text_1, emb_text_2 = self.embed_text_pairs(text1, text2)
        return self.get_similarity_score(emb_text_1, emb_text_2)
