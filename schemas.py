from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options")
    answer: str = Field(description="The correct option, copied exactly from options")
    explanation: str = Field(description="One or two sentences explaining why")


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]
