# fine-tuning-example

## Class Setup

### Setup LangSmith

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=fine-tuning-example 
```

### Building training set (at least train the model before class)

1. Run the `main.py` class with the gpt-3.5-turbo model. Make sure the system message is in the prompt. This will produce 60 messages. 
1. In the `fine_tune-model.ipynb`, you can execute all the code to get the 60 messages, save them to a dataset, and use that dataset to fine-tune a model (which will take about 10 minutes)
1. Once the fine-tuned model is created, you can use it in the main.py file and remove the system message.
1. Use the `evaluation.py` class to test thhe gpt-3.5-turbo model against the newly fine-tuned model. You can go into the dataset section and see how the two compare/

## Optional steps

### Launch LangServe

```bash
langchain serve
```

### Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

#### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

#### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
