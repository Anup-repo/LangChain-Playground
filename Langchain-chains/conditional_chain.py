from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI()

# Define the output parser
class Feedback(BaseModel):
    feedback: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback, either 'positive' or 'negative'."
    )

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

# This parser will convert the LLM's output into a Pydantic model, ensuring the output is structured correctly.


# Define the prompt template
prompt_template = PromptTemplate(
    template="Classify the sentiment of the following text as positive or negative. \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)


classifier_chain = prompt_template | llm | pydantic_parser


# Define the parallel chain
prompt1 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)

# Define output_parser
output_parser = StrOutputParser()

# Define the branch chain that processes both positive and negative feedback
branch_chain = RunnableBranch(
    (lambda x: x.feedback == "positive", prompt1 | llm | output_parser),
    (lambda x: x.feedback == "negative", prompt2 | llm | output_parser),
    RunnableLambda(lambda x: "Could find sentiment of the feedback.")
)

chain = classifier_chain | branch_chain

# Run the chain with an example input
output = chain.invoke({"feedback": "I hate this product!"})

print(f"{output}")

# Print the graph structure
chain.get_graph().print_ascii()


