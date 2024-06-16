from transformers import pipeline

def get_answer_from_model(context, question):
    """Gets an answer from a pre-trained model based on the input context and question."""
    # Load the pre-trained model for question answering
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    # Use the model to answer the question based on the context
    result = qa_pipeline(question=question, context=context)
    
    return result['answer']

