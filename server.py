from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from app import app as grading_graph

#initialize the application
server = FastAPI(title = "AI Grading App", version = "1.0")

#ensure temporary directory exists for uploaded files
os.makedirs("temp_uploads",exist_ok = True)

@server.post("/grade_exam/")
async def grade_exam_endpoint(
    student_id: str = Form(...),
    groq_api_key: str = Form(...),
    question_paper: UploadFile = File(None),
    answer_key: UploadFile = File(None),
    student_copy: UploadFile = File(None)
):
    print(f"\n[API] Recieved grading request for student: {student_id}")

    #save uploaded file to disk for langchain to read it
    file_paths = {
        "question_paper_path":None,
        "answer_key_path":None,
        "student_copy_path":None
    }
    async def save_upload(upload_file: UploadFile):
        if not upload_file:
            return None
        # Guarantee the folder exists right before saving
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = f"temp_uploads/{uuid.uuid4()}_{upload_file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return file_path

    file_paths["question_paper_path"] = await save_upload(question_paper)
    file_paths["answer_key_path"] = await save_upload(answer_key)
    file_paths["student_copy_path"] = await save_upload(student_copy)

    # 2. Prepare the LangGraph payload
    # We use the student_id as the thread_id so SQLite remembers this specific student!
    config = {'configurable': {'thread_id': student_id}}
    
    initial_input = {
        "groq_api_key": groq_api_key,
        "question_paper_path": file_paths["question_paper_path"],
        "answer_key_path": file_paths["answer_key_path"],
        "student_copy_path": file_paths["student_copy_path"],
        "raw_question_text": None,
        "raw_answer_key_text": None,
        "raw_student_text": None
    }

    # 3. Trigger your backend LangGraph pipeline
    try:
        final_state = grading_graph.invoke(initial_input, config)
        
        # 4. Clean up temporary files so your server doesn't run out of storage
        for path in file_paths.values():
            if path and os.path.exists(path):
                os.remove(path)

        # 5. Return the JSON to the frontend
        return JSONResponse(content={
            "student_id": student_id,
            "total_score": final_state.get("total_score"),
            "grades": final_state.get("final_grades")
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8000)