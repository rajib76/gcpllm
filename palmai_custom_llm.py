# The PALMAPI abstraction in LANGCHAIN does not seem to be complete yet. It is there
# in their github, but no documentation yet. Also when I tried to use it, got an error from GCP
# as shown below. So, it looks like that GCP also did not roll it out yet to public

# [links { description: "Google developers console API activation" url:
# "https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=183578915041" }
# , reason: "SERVICE_DISABLED" domain: "googleapis.com" metadata { key: "service" value:
# "generativelanguage.googleapis.com" } metadata { key: "consumer" value: "projects/183578915041" } ]

# It seems like the Palm API is only available through the Google SDK, hence went ahead and created a
# custom LLM to wrap it with langchain

import os
from typing import Any, List, Optional, Dict

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Extra, root_validator


class PalmAI(LLM):
    client: Any  #: :meta private:
    model: Optional[str] = "text-bison@001"
    gcp_apl_cred: Optional[str] = None
    gcp_project: Optional[str] = None
    location: str = "us-central1"
    tuned_model_name: str = ""
    temperature: int =0
    max_output_tokens: int = 256
    top_k: int = 1
    top_p: int = 1
    stop: Optional[List[str]] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        gcp_apl_cred = get_from_dict_or_env(values, "gcp_apl_cred", "GOOGLE_APPLICATION_CREDENTIALS")
        values["gcp_apl_cred"] = gcp_apl_cred
        gcp_project = get_from_dict_or_env(values, "gcp_project", "GCP_PROJECT")
        values["gcp_project"] = gcp_project
        return values

    @property
    def _llm_type(self) -> str:
        return "palmai"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              ) -> str:

        import vertexai
        from vertexai.preview.language_models import TextGenerationModel

        vertexai.init(project=self.gcp_project, location=self.location)
        model = TextGenerationModel.from_pretrained(self.model)

        params = self._default_params
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            params["stop_sequences"] = self.stop
        else:
            params["stop_sequences"] = stop

        if self.tuned_model_name:
            model = model.get_tuned_model(self.tuned_model_name)

        response = model.predict(
            prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_k=self.top_k,
            top_p=self.top_p, )


        return response.text
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Cohere API."""
        return {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}


if __name__ == "__main__":
    load_dotenv()
    gcp_apl_cred= os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_project = os.getenv("GCP_PROJECT")
    pm = PalmAI()
    pm.gcp_project=gcp_project
    pm.gcp_apl_cred=gcp_apl_cred
    responses = pm.generate(["who is indira gandhi"])
    generations = responses.generations[0]
    print(generations[0].text)
