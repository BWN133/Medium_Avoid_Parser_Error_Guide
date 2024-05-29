from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
class AnimalOutput(BaseModel):
    """Output format of animal description analysis"""
    reasoning:str = Field(..., description="Step by step analysis of the input animal description")
    result: str = Field(..., description="Your final answer of the question. Should be either a 'Dog', 'Cat', or 'Turtle'.")

if __name__ == '__main__':
    load_dotenv() # Load your OpenAI API environment variables
    parser = PydanticOutputParser(pydantic_object=AnimalOutput)
    prompt = ChatPromptTemplate.from_messages([
    (
        "You are an expert in identifying animals. "
        "You will be provided with a description of an animal. "
        "Your job is to analyze the description step-by-step and determine if the animal is a cat, a dog, or a turtle. "
        "Remember, concepts from two tags should be included. "
        "You will be outputting a json"
        "Here is the input description: {description}"
    )
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    
    # Actual LLM Chain
    runnable = prompt | llm | parser
    test_description = "This small, agile creature is known for its grace, independent spirit, and penchant for napping in sunbeams."
    print(runnable.invoke({ "description": test_description}))
 
