from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the LLM
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)


# Define the prompt template
prompt_template = PromptTemplate(
    template="Give a detailed report on {topic}?", 
    input_variables=["topic"]
)

# Define second prompt template

second_prompt_template = PromptTemplate(
    template="Summarize the report on {topic} in two sentences.", 
    input_variables=["topic"]
)

# Create the output parser

output_parser = StrOutputParser()

# Define the chain

chain = prompt_template | model | output_parser | second_prompt_template | model | output_parser

# Run the chain with an example input
topic = "Artificial Intelligence"
output = chain.invoke({"topic": topic})

print(f"{output}")

# Print the graph structure
chain.get_graph().print_ascii()
