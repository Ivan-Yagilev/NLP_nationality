from fastapi import FastAPI
from use import predict
from models import Input

app = FastAPI()

@app.get("/")
def nationality(input: Input):
    return predict(input.surname)