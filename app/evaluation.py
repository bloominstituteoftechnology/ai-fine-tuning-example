from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langsmith.wrappers import wrap_openai
from openai import Client as OpenAIClient

# Define dataset: these are your test cases
dataset_name = "Fine-Tuning Dataset Example"

openai_client = wrap_openai(OpenAIClient())

# Define predictor functions
def predict_with_gpt_3_5_turbo(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs['input'][1]['data']['content']}]
    response = openai_client.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
    return {"output": response}

def predict_with_fine_tuned_model(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs['input'][1]['data']['content']}]
    response = openai_client.chat.completions.create(messages=messages, model="ft:gpt-3.5-turbo-0125:xevant::9Q0Hk9G4")
    return {"output": response}

# Define evaluators
def check_for_pirate_talk(run: Run, example: Example) -> dict:
    llm = ChatOpenAI()
    prompt = PromptTemplate(template="Determine if the following response sounds like a pirate: {response}\n\nscore a 1 if it does and a 0 if it doesn't. Only return the score.", input_variables=["response"])
    chain = prompt | llm | StrOutputParser()
    run_output = run.outputs['output'].choices[0].message.content
    response = chain.invoke({"response": run_output})
    return {"key":"check_for_pirate_talk", "score": 1 if "1" in response else 0}

baseline_experiment_results = evaluate(
    predict_with_gpt_3_5_turbo, # predictor
    data=dataset_name, # The data to predict and grade over
    evaluators=[check_for_pirate_talk], # The evaluators to score the results
    experiment_prefix="pirate-sounding", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
)

fine_tuned_experiment_results = evaluate(
    predict_with_fine_tuned_model, # predictor
    data=dataset_name, # The data to predict and grade over
    evaluators=[check_for_pirate_talk], # The evaluators to score the results
    experiment_prefix="pirate-sounding", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
)