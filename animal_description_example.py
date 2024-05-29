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
    load_dotenv() # Load your OpenAI 
    parser = PydanticOutputParser(pydantic_object=AnimalOutput)
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "You are an expert in identifying animals"
            "You will be provided with a description of an animal"
            "Your job is to analyze the description step by step and predict whether the animal is a cat, a dog, or a turtle"
            "Remenber, concepts from two tag should be included"
            "You will be outputing a dictionary following format: "
            "{output}"
            "Remenber to use double quote \" for key values"
            "Here is the input description: {description}"
        )
    ]
    ).partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    outputformat = """{"reasoning":"<content>","result":"<content>"}"""
    runnable = prompt | llm | parser
    test_description = "This four-legged furry companion is known for its loyalty and wagging tail."
    runnable.invoke({"output":outputformat, "description":test_description })
 
