import pydantic

class Input(pydantic.BaseModel):
    surname: str

class Output(pydantic.BaseModel):
    prediction: str
    score: float