from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import login
from langchain.schema.runnable import RunnableParallel


load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))  # Login to Hugging Face

# Initialize the LLM

model1 = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)


llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")

model2 = ChatHuggingFace(llm=llm)

# Define the first prompt template

template1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

# Define the second prompt template
template2 = PromptTemplate(
    template= "Generate 5 quiz questions and answers from the following text \n {text}",
    input_variables=['text']
)

# Define the third prompt template
template3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

# Create the output parser
output_parser = StrOutputParser()

# Define the parallel chain

parallel_chain = RunnableParallel({
    "notes": template1 | model2 | output_parser,
    "quiz": template2 | model2 | output_parser
    }
)

# define the merge chain
merge_chain = template3 | model1 | output_parser

# Define the final chain 
final_chain = parallel_chain | merge_chain

# Run the final chain with an example input
text = "Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction. AI applications include expert systems, natural language processing, speech recognition, and machine vision."

output = final_chain.invoke({"text": text})

print(f"{output}")

# Print the graph structure
final_chain.get_graph().print_ascii()

