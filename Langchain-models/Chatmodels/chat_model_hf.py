from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))  # Login to Hugging Face

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)
