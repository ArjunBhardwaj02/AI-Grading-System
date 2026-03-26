from pydantic import BaseModel, Field
from typing import List

#structuring questions and their answers
#Defining the single question
class QuestionSchema(BaseModel):
    question_number: str = Field(description="The question number or ID (e.g., '1', '2a', 'Q3')")
    question_text: str = Field(description="The actual text of the question asked")
    answer_key: str = Field(description="The official answer or marking scheme for this specific question")
    max_marks: float = Field(default=1.0,description="Maximum marks available. Use float in case of half marks like 2.5")

# Define the Exam as a LIST of those single questions
class ExamSchema(BaseModel):
    questions: List[QuestionSchema] = Field(description="A list containing all the structured questions from the exam paper")


#structuring messy student answer

class StudentSchema(BaseModel):
    question_number : str = Field(description="The question number or ID (e.g., '1', '2a', 'Q3')")
    student_answer : str = Field(description="Answer of the student for that question")

class StudentAnswerSchema(BaseModel):
    answers : List[StudentSchema] = Field(description="A list containing all the structured answers of questions in exam by student")





#defining the grading structure

class grade_structure(BaseModel):
    question: str = Field(description="Question asked in exam")
    answer: str = Field(description="Answer written by student")
    marks_awarded: float = Field(description="The numeric marks awarded to the student for this specific question")
    justification: str = Field(description="Brief reason for why these marks were awarded based on the key")

from typing import TypedDict, List, Optional

class GradingState(TypedDict, total = False):
    #api
    groq_api_key: str
    #input from teacher
    question_paper_path : Optional[str]
    answer_key_path : Optional[str]
    student_copy_path:Optional[str]

    #extracted raw text (From OCR) , optional as if user just type
    raw_question_text : Optional[str]
    raw_answer_key_text : Optional[str]
    raw_student_text : Optional[str]

    #structured data
    structured_exam_data: ExamSchema
    structured_student_data : StudentAnswerSchema

    #final output
    final_grades : List[dict]   #list of question, answer, grades
    total_score : float
    error: Optional[str]