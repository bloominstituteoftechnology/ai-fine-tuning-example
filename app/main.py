from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from training_set import *
from functools import partial
import concurrent.futures

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:bloomtech::9SCaCLTS") # This is the fine-tuned model. You can remove the system message when using this model.

def generate_review_responses(prompt_tag, reviews):
    for review in reviews:
        prompt = ChatPromptTemplate(
            tags=["reviews", prompt_tag], 
            messages=[
                # SystemMessage(content="You are a customer service agent that talks in a pirate voice"), # Comment this line if using the fine-tuned model
                HumanMessage(content=f"Provide a customer-facing response for the following review: {review}")
            ],
        )
        chain = prompt | llm
        chain.invoke({})


generate_positive_responses = partial(generate_review_responses, prompt_tag="positive", reviews=positive)
generate_negative_responses = partial(generate_review_responses, prompt_tag="negative", reviews=negative)
generate_neutral_responses = partial(generate_review_responses, prompt_tag="neutral", reviews=neutral)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(lambda func: func(), [generate_positive_responses, generate_negative_responses, generate_neutral_responses])

