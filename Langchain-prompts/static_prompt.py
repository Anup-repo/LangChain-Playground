from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temprature=0.5, max_tokens=100)

result = llm.invoke("Summarize `Attention All you Need` research paper in 5 lines.")

print(result.content)

print("Metadata:", result.usage_metadata)
