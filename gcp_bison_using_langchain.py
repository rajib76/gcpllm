import os

from dotenv import load_dotenv
from langchain.llms import GooglePalm

# Got below error
# [links { description: "Google developers console API activation" url:
# "https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=183578915041" }
# , reason: "SERVICE_DISABLED" domain: "googleapis.com" metadata { key: "service" value:
# "generativelanguage.googleapis.com" } metadata { key: "consumer" value: "projects/183578915041" } ]

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

llm = GooglePalm()
response = llm("Who is Indira Gandhi")

print(response)
