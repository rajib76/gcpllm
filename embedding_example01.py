import os

from dotenv import load_dotenv
from vertexai.preview.language_models import TextEmbeddingModel


load_dotenv()
gcp_apl_cred= os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
gcp_project = os.getenv("GCP_PROJECT")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_apl_cred
os.environ["GCP_PROJECT"] = gcp_project

def text_embedding():
  """Text embedding with a Large Language Model."""
  model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
  embeddings = model.get_embeddings(["What is life?"])
  print(embeddings)
  for embedding in embeddings:
      print(embedding.values)
  #     vector = embedding.values
  #     print(vector)
  #     print(f'Length of Embedding Vector: {len(vector)}')

if __name__=="__main__":
    text_embedding()