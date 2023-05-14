from abc import abstractmethod
import numpy as np


class LLMEmbed():
    def __init__(self):
        self.module = "Embedding"

    @abstractmethod
    def embed_text_pairs(self,text1,text2):
        pass

    @abstractmethod
    def get_score(self,text1, text2):
        pass

    @classmethod
    def get_similarity_score(cls,emb1,emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
