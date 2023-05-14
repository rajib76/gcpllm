import os

from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings

from embedding.llm_embed import LLMEmbed

load_dotenv()
API_KEY = os.getenv("openai_api_key")


class Embedding(LLMEmbed):
    def __init__(self, model_id):
        super().__init__()
        self.module = "OpenAI"
        openai = OpenAIEmbeddings(model=model_id, openai_api_key=API_KEY)
        self.embed_model = openai

    def embed_text_pairs(self, text1="", text2=""):
        print("Embedding with {model}:".format(model=self.module))
        emb_text_1 = self.embed_model.embed_query(text1)
        emb_text_2 = self.embed_model.embed_query(text2)

        return emb_text_1,emb_text_2

    def get_score(self,text1, text2):
        emb_text_1, emb_text_2 = self.embed_text_pairs(text1, text2)
        return self.get_similarity_score(emb_text_1, emb_text_2)



if __name__ == "__main__":
    cemb = Embedding()
    cemb.get_similarity_score()
