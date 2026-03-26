
from state import GradingState, ExamSchema, StudentAnswerSchema, grade_structure

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import os


from vision import transcribe_exam_paper, process_pdf_document


def ocr_node(state: GradingState):
    print('[System] OCR Node: Scanning images: \n')
    user_api_key = state.get("groq_api_key")

    state_updates = {}

    #A temporary dictionary to remember files we already processed in this run
    extraction_cache = {}

    def smart_extract(file_path):
        """Helper to route to the correct extractor based on extension"""
        # Strip the unique UUID prefix from the filename to use as the cache key
        # UUIDs are 36 chars + 1 underscore = 37 chars. We split by the first '_'
        cache_key = os.path.basename(file_path).split('_', 1)[-1] if '_' in os.path.basename(file_path) else file_path


        # If we already extracted this file, just return the saved text instantly
        if cache_key in extraction_cache:
            print(f"-> [Cache Hit] Skipping API call, reusing text for {cache_key}")
            return extraction_cache[cache_key]

        if file_path.lower().endswith('.pdf'):
            result = process_pdf_document(file_path,user_api_key)
        else:
            with open(file_path, "rb") as f:
                result = transcribe_exam_paper(f,user_api_key)
        extraction_cache[cache_key] = result
        return result

    question_path = state.get('question_paper_path')
    if question_path and os.path.exists(question_path):
        print('transcribing question paper')
        state_updates["raw_question_text"] = smart_extract(question_path)
    
    student_img = state.get('student_copy_path')
    if student_img and os.path.exists(student_img):
        print("-> Transcribing Student Answer Sheet...")
        state_updates["raw_student_text"] = smart_extract(student_img)

    
    answer_key_img = state.get('answer_key_path')
    if answer_key_img and os.path.exists(answer_key_img):
        print("-> Transcribing Answer Key...")
        state_updates["raw_answer_key_text"] = smart_extract(answer_key_img)

    if not state_updates:
        print("[Warning] OCR Node ran, but no valid image paths were found in the state.")

    return state_updates


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


def structure_node(state:GradingState):
    user_api_key = state.get("groq_api_key")
    raw_question_text = state['raw_question_text']
    raw_answer_key = state['raw_answer_key_text']
    model = ChatGroq(
        model = "llama-3.1-8b-instant",
        api_key=user_api_key,
        temperature=0 
    )
    system_prompt = (
        "You are a precise data extraction assistant. Your job is to take the raw, messy OCR text "
        "of an exam question paper and its corresponding answer key, and structure it perfectly. "
        "Ensure you match the correct answer key and max marks to the correct question. Do not leave out any questions. "
        "CRITICAL: The 'question_number' MUST be purely the numeric digit (e.g., '1', '2'). Strip away 'Q', 'Ans', or punctuation."
    )

    prompt = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        ('human','Raw Question Paper: \n{raw_questions}\n\nRaw Answer key: \n{raw_answer_key}')
    ])

    new_model = model.with_structured_output(ExamSchema)
    chain = prompt | new_model
    response = chain.invoke({
        'raw_questions':raw_question_text,
        'raw_answer_key':raw_answer_key
    })

    return {"structured_exam_data":response}


def student_structure_node(state:GradingState):
    user_api_key = state.get("groq_api_key")
    model = ChatGroq(
        model = "llama-3.1-8b-instant",
        api_key=user_api_key,
        temperature=0 
    )
    raw_student_text = state['raw_student_text']
    raw_question_text = state['raw_question_text']

    system = (
        "You are a precise data extraction assistant structuring student answers. "
        "WARNING: Students frequently skip questions! Do not just number answers 1, 2, 3 sequentially. "
        "You MUST cross-reference the student's text with the Exam Questions to deduce the correct 'question_number'. "
        "For example, if the student's 3rd written answer matches the topic of Question 4, assign it 'question_number': '4'. "
        "CRITICAL: The 'question_number' MUST be purely the numeric digit (e.g., '1', '2')."
    )

    prompt = ChatPromptTemplate.from_messages([
        ('system',system),
        ('human', 'Exam Questions Context:\n{questions}\n\nStudent Answer OCR:\n{student_answer}')
    ])
    
    model = model.with_structured_output(StudentAnswerSchema)
    chain = prompt| model 

    response = chain.invoke({"student_answer":raw_student_text,"questions": raw_question_text})
    return {"structured_student_data":response}

def grade_node(state:GradingState):
    user_api_key = state.get("groq_api_key")

    print('[System] Grade Node: Batching questions for parallel execution...\n')
    structured_exam_data = state['structured_exam_data']
    structured_student_data = state['structured_student_data']

    system = (
        "You are an expert, fair invigilator grading an exam. "
        "Grade the student's answer against the answer key. "
        "CRITICAL GRADING RULES: "
        "1. Do not expect word-for-word exact matches. Grade based on semantic meaning, core concepts, and keywords. "
        "2. If the student captures the core essence of the answer, award full marks. "
        "3. You MAY award partial marks (e.g., 1.5 out of 3.0) if the answer is incomplete but partially correct. "
        "4. If the student answer is exactly 'No answer provided.', award 0.0. "
        "5. The 'marks_awarded' field MUST be a raw float (e.g., 1.0, 0.5, 0.0) and must not exceed the max marks available. DO NOT wrap it in quotes. "
        "Provide a brief justification explaining why you awarded these specific marks."
    )

    prompt = ChatPromptTemplate.from_messages([
        ('system',system),
        ('human',"question: {question}\n max_marks_available: {max_marks}\n answer_key: {answer_key}\n student_answer: {student_answer}")
    ])

    model = ChatGroq(
        model = "llama-3.3-70b-versatile",
        api_key=user_api_key,
        temperature=0.6 # as student can have some creativity writing answer
    )

    structed_llm = model.with_structured_output(grade_structure)

    chain = prompt | structed_llm

    #prepare the payload list instead of firing requests instantly
    batch_payload = []
    for q in structured_exam_data.questions:
        specific_student_answer = "No answer provided." # Default fallback
        for student_ans in structured_student_data.answers:
            if q.question_number == student_ans.question_number:
                specific_student_answer = student_ans.student_answer
                break
        
        batch_payload.append({
            "question":q.question_text,
            "max_marks": q.max_marks,
            "answer_key":q.answer_key,
            "student_answer":specific_student_answer
        })

    print(f"-> Blasting {len(batch_payload)} questions to Llama 70B concurrently...")
    responses = chain.batch(batch_payload) #creating response in batches

    l =[]
    calculate_total_score =0.0

    for response in responses:
        grade_dict = {
            "question":response.question,
            "answer":response.answer,
            "marks_awarded":response.marks_awarded,
            "justification":response.justification
        }
        l.append(grade_dict)
        calculate_total_score += response.marks_awarded
    return {'final_grades':l,"total_score":calculate_total_score}
       


def question_text_or_img(state:GradingState):
    if any([state.get('question_paper_path'), state.get('student_copy_path'), state.get('answer_key_path')]):
        return "image"
    else:
        return "text"


from langgraph.graph import START, END, StateGraph
workflow = StateGraph(GradingState)
# 'check_same_thread=False' is critical for FastAPI/Streamlit later
conn = sqlite3.connect("grades_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

workflow.add_node("ocr_node",ocr_node)
workflow.add_node("grade_node",grade_node)
workflow.add_node('structure_node',structure_node)
workflow.add_node('student_structure_node',student_structure_node)

workflow.add_conditional_edges(START, question_text_or_img, {"image":"ocr_node", "text":"structure_node"})
workflow.add_edge("ocr_node","structure_node")
workflow.add_edge('structure_node','student_structure_node')
workflow.add_edge("student_structure_node",'grade_node')
workflow.add_edge("grade_node",END)

app = workflow.compile(
    checkpointer=memory,
)
