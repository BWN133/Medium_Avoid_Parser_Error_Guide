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

def build_few_shot_examples(description:str, reasoning:str, result:str):
    result = "#####description: " + description + "#####\n\n"
    
    d = """{"reasoning":""" +"\""+ reasoning+"\""+ ""","result":""" +"\""+ result + """\"}"""
    return result + d
def exception_to_messages(inputs: dict, VERBOSE=True) -> dict:

    exception = str(inputs['exception'])
    # Add historical messages to the original input, so the model knows that it made a mistake with the last tool call.
    messages = ChatPromptTemplate.from_messages ([
        "The last call raised an exception:",
        exception,
        "Do not repeat mistakes and try again."
    ])
    inputs["last_output"] = messages
    return input


if __name__ == '__main__':
    load_dotenv() # Load your OpenAI API environment variables
    parser = PydanticOutputParser(pydantic_object=AnimalOutput)
    prompt = ChatPromptTemplate.from_messages([
    (
        "You are an expert in identifying animals. "
        "You will be provided with a description of an animal. "
        "Your job is to analyze the description step-by-step and determine if the animal is a cat, a dog, or a turtle. "
        "Remember, concepts from two tags should be included. "
        "You will be outputting a dictionary in the following format: "
        "{output} "
        "Few-shot examples will have system prompts encapsulated in the delimiter '#####', and your output should not include the '#####' delimiters. "
        "Here are the few-shot examples: "
        "{examples} "
        "Remember to use double quotes \" for key values. "
        "Here is the input description: {description}"
    )
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    # Build Output Template
    outputformat = """{"reasoning":"<content>","result":"<content>"}"""

    # Build Few Shot examples
    example = build_few_shot_examples(
        description="This four-legged furry companion is known for its loyalty and wagging tail.",
        reasoning="The description mentions a four-legged furry companion known for loyalty and a wagging tail. Cats and turtles are typically not associated with wagging tails, hence the animal is most likely a dog.",
        result="dog")
    
    # Actual LLM Chain
    runnable = prompt | llm | parser
    self_correct_enhance_chain = runnable.with_fallbacks([exception_to_messages | runnable], exception_key="exception")
    test_description = "This small, agile creature is known for its grace, independent spirit, and penchant for napping in sunbeams."
    print(self_correct_enhance_chain.invoke({"output": outputformat, "description": test_description, "examples":example}))
 
