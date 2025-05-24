from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from huggingface_hub import login
import os
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))  # Login to Hugging Face

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me random name, age and location of a fictional person.\n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Use format function to fit the partial variables to the template and generate the prompt
print(template.format())

response = model.invoke(template.format())

print(parser.parse(response.content))

print("=====================================================Using JSONParser================================================")
chain = template | model | parser

result = chain.invoke({})

print(result)