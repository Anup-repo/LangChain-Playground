from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


load_dotenv()

model = ChatOpenAI()

class Student(BaseModel):

    name: str = Field(default="John Doe", description="The name of the student")
    age: Optional[int] = Field(default=None, description="The age of the student")
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of the student",
    )

structure_model = model.with_structured_output(Student)

student_description = """
John Doe is a dedicated student with a passion for learning. At the age of 20, he has already achieved a commendable CGPA of 8.5. His email, johndoe@example.com, is a testament to his professional approach. John enjoys hobbies such as reading and playing soccer, and he aspires to become a software engineer in the future.
"""

result = structure_model.invoke(student_description)

print(result)
print(result["name"])
print(result["age"])
print(result["email"])
print(result["cgpa"])
