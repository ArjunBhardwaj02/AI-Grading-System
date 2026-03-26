import streamlit as st
import requests

# Set page config for a wider, cleaner layout
st.set_page_config(page_title="AI Exam Grader", layout="wide")

# The URL of your FastAPI backend
API_URL = "http://localhost:8000/grade_exam/"

st.title("📝 Automated AI Exam Grader")
st.markdown("Upload the exam materials and the student's answer sheet to instantly generate a graded report.")

with st.sidebar:
    st.header("🔑 API Configuration")
    user_api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("[Get your free Groq API key here](https://console.groq.com/keys)")

# Create a clean UI layout using columns
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Exam Materials")
    student_id = st.text_input("Student ID (e.g., student_001)", placeholder="Enter unique student ID...")
    
    question_paper = st.file_uploader("Upload Question Paper (PDF/Image)", type=['pdf', 'jpg', 'jpeg', 'png'])
    answer_key = st.file_uploader("Upload Answer Key (PDF/Image)", type=['pdf', 'jpg', 'jpeg', 'png'])

with col2:
    st.header("2. Upload Student Work")
    st.info("Ensure the image is clear and well-lit.")
    student_copy = st.file_uploader("Upload Student Answer Sheet (Image/PDF)", type=['pdf', 'jpg', 'jpeg', 'png'])

st.divider()

# The Execution Button
if st.button("🚀 Grade Exam", use_container_width=True, type="primary"):
    if not user_api_key: # <--- Block execution if no key is provided
        st.error("🚨 Please enter your Groq API Key in the sidebar.")
    # Input Validation
    elif not student_id:
        st.error("Please enter a Student ID.")
    elif not student_copy:
        st.error("Please upload the Student Answer Sheet.")
    else:
        with st.spinner(f"Analyzing and grading paper for {student_id}... Please wait."):
            try:
                # Prepare the files to send to FastAPI
                files = {}
                
                # Only attach files if the user actually uploaded them
                if question_paper:
                    files['question_paper'] = (question_paper.name, question_paper.getvalue(), question_paper.type)
                if answer_key:
                    files['answer_key'] = (answer_key.name, answer_key.getvalue(), answer_key.type)
                
                files['student_copy'] = (student_copy.name, student_copy.getvalue(), student_copy.type)

                # Prepare the text form data
                data = {'student_id': student_id,'groq_api_key': user_api_key}

                # Send the POST request to FastAPI
                response = requests.post(API_URL, data=data, files=files)

                # Handle the Response
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Grading Complete!")
                    st.metric(label="Total Score Awarded", value=f"{result['total_score']} Marks")
                    
                    st.subheader("Detailed Breakdown")
                    
                    # Display each graded question in a clean expander
                    for grade in result['grades']:
                        with st.expander(f"Question: {grade['question']} — Score: {grade['marks_awarded']}"):
                            st.markdown(f"**Student's Answer:** {grade['answer']}")
                            st.markdown(f"**AI Justification:** {grade['justification']}")
                else:
                    st.error(f"Backend Error: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend server. Is FastAPI running?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")