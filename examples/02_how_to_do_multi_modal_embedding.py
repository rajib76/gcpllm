import os
from operator import itemgetter
from typing import List

import numpy as np
from dotenv import load_dotenv
from vertexai.vision_models import Image, MultiModalEmbeddingModel

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

def get_similarity_score(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def get_multimodal_embedding(image_path,
                             contextual_text):

    image = Image.load_from_file(image_path)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408,
    )
    img_embedding = embeddings.image_embedding
    txt_embedding = embeddings.text_embedding

    return img_embedding, txt_embedding
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")


def create_vector_index(image_list: List):
    vectors = []
    vector = {}
    for image in image_list:
        image_name = image["name"]
        print(image_name)
        image_path = image["image_path"]
        print(image_path)
        image_contextual_text = image["contextual_text"]
        img_embedding, txt_embedding = get_multimodal_embedding(image_path, image_contextual_text)
        vector = {"img_embed": img_embedding, "txt_embed": txt_embedding, "image_name": image_name}
        vectors.append(vector)
        vector = {}

    return vectors

def search_image_with_text(vectors,text):
    embeddings = model.get_embeddings(
    contextual_text=text,
    dimension=1408)

    txt_embedding = embeddings.text_embedding

    for vector in vectors:
        vector_img_embedding = vector["img_embed"]
        score = get_similarity_score(txt_embedding,vector_img_embedding)
        vector["score"] = score

    return vectors



if __name__ == "__main__":
    image_list = [
        {"name": "kitchen",
         "image_path": "/Users/joyeed/gcpexample/gcpllm/data/kitchen.jpeg",
         "contextual_text": "Image of a kitchen"},
        {"name": "sunset",
         "image_path": "/Users/joyeed/gcpexample/gcpllm/data/sunset.jpeg",
         "contextual_text": "A beautiful sunset"}

    ]

    vectors = create_vector_index(image_list=image_list)
    print(vectors)

    text = "Image of a kitchen"
    mod_vectors = search_image_with_text(vectors=vectors,text=text)
    mod_vectors=sorted(mod_vectors, key=itemgetter('score'), reverse=True)
    for vector in mod_vectors:
        print(vector["image_name"],":",vector["score"])
