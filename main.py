from fastapi import FastAPI

import services.service

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    # Input data
    input_data = """name: Sarthak
    age: 30
    roll no.: 202001
    marks: 78/100"""

    # Input question
    question = "what is the roll no?"

    # Get the answer from the model
    answer = services.service.get_answer_from_model(input_data, question)
    print("_____________________________")
    print(answer)
