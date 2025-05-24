from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=1024)

paper_input = input("Enter the title of the research paper: ")
style_input = input(
    "Enter the explanation style(Beginner-Friendly",
    "Technical",
    "Code-Oriented",
    "Mathematical): ",
)
length_input = input(
    "Enter the explanation length(Short (1-2 paragraphs)",
    "Medium (3-5 paragraphs)",
    "Long (detailed explanation)): ",
)

template = """
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
   - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
prompt_template = PromptTemplate(
    template=template,
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True,
)

# if you have any saved template files, you can load them using the load_prompt function
# prompt_template = load_prompt("template.json")

prompt = prompt_template.invoke(
    {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    }
)

result = llm.invoke(prompt)

print(result.content)


# Instead of using two invoke function one for the template and one for the llm, we can use chain

chain = prompt_template | llm

result = chain.invoke(
    {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
    }
)

print(result.content)
