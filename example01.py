# Plain invokation without any framework
import os

import vertexai
from dotenv import load_dotenv
from vertexai.preview.language_models import TextGenerationModel

load_dotenv()
gcp_apl_cred= os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
gcp_project = os.getenv("GCP_PROJECT")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_apl_cred
os.environ["GCP_PROJECT"] = gcp_project


def predict_large_language_model_sample(
        project_id: str,
        model_name: str,
        temperature: float,
        max_decode_steps: int,
        top_p: float,
        top_k: int,
        content: str,
        location: str = "us-central1",
        tuned_model_name: str = "",
):
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
        model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p, )
    print(f"Response from Model: {response.text}")


predict_large_language_model_sample("uplifted-woods-383322", "text-bison@001", 0.2, 256, 0.8, 40, '''Who is Indira Gandhi''', "us-central1")