from langchain_core.prompts import ChatPromptTemplate

# Here SystemMessage and HumanMessage doesnot work as compared to PromptTemplate, use System and Human instead
chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert"),
        ("human", "Explain in simple terms, what is {topic}"),
    ]
)

prompt = chat_template.invoke({"domain": "cricket", "topic": "Dusra"})

print(prompt)
