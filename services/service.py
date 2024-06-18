from transformers import pipeline
from bs4 import BeautifulSoup

# we are assuming that the html is given with the request. 
def get_context_from_html(html):
    "create a context for product using given html"
    # TODO - we should use llm to create context from html
    # TODO - llm output will be key value pair, which used to give user's query answer.
    pass

def get_answer_from_model(context, question):
    """Gets an answer from a pre-trained model based on the input context and question."""
    # Load the pre-trained model for question answering
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    # Use the model to answer the question based on the context
    result = qa_pipeline(question=question, context=context)
    
    return result['answer']

