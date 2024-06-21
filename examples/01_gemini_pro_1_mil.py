import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
result = llm.invoke("Write a ballad about LangChain")
print(result.content)