from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI()

# Define the prompt template
prompt_template = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"]
)

# Create the output parser
output_parser = StrOutputParser()

# Define the chain
result = prompt_template | llm | output_parser

# Run the chain with an example input
country = "France"
output = result.invoke({"country": country})

print(f"{output}")

result.get_graph().print_ascii()
