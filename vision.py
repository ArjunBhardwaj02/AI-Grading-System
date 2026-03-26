import base64
import os
from io import BytesIO
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import fitz #pymupdf
from groq import Groq

load_dotenv()

def process_pdf_document(pdf_path, api_key:str):
    """Slices a PDF into individual pages, transcribes each page using the Vision model,
    and stitches the extracted text together."""

    print(f"\n[PDF Slicer] Opening {pdf_path}...")

    try:
        doc = fitz.open(pdf_path)
        full_transcription = ""

        #loop through every page
        for page_num in range(len(doc)):
            print(f"->Extracting Page {page_num+1} of {len(doc)}...")
            page = doc.load_page(page_num)
            #render the page
            pix = page.get_pixmap(dpi=150)

            img_bytes = pix.tobytes('jpeg')
            image_stream = BytesIO(img_bytes)

            page_text = transcribe_exam_paper(image_stream, api_key)
            if page_text:
                full_transcription += f"\n---PAGE {page_num+1} ---\n {page_text}"
            else:
                print(f'[Warning] Failed to extract text from page {page_num +1}')
        return full_transcription.strip()
    except Exception as e:
        print(f'[Critical Error] Failed to process the PDF: {e}')
        return None




def get_vision_model(api_key:str):
    """Initializes the Groq Vision model."""
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=api_key,
        temperature=0 
    )

def auto_rotate_image(img, model):
    """
    Acts as an Orientation Agent. Asks the Vision API for the rotation angle 
    and physically rotates the Pillow image object before transcription.
    """
    print("Checking image orientation...")
    
    # 1. Create a tiny, low-quality thumbnail just for checking rotation (saves API bandwidth)
    check_img = img.copy()
    check_img.thumbnail((400, 400))
    buffer = BytesIO()
    check_img.save(buffer, format="JPEG", quality=50)
    b64_thumb = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 2. Ask the model exactly ONE question
    prompt = """
    Look at the handwritten text in this image. What is its orientation?
    Respond with ONLY ONE of the following numbers:
    0 (Text is upright and readable normally)
    90 (Text is rotated 90 degrees clockwise)
    180 (Text is upside down)
    270 (Text is rotated 90 degrees counter-clockwise)
    Do not output any words, punctuation, or explanations. Only the number.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_thumb}"}}
        ]
    )

    try:
        response = model.invoke([message])
        angle_str = response.content.strip()
        
        # 3. Rotate the ORIGINAL high-res image based on the AI's answer
        if angle_str == "90":
            print("-> Detected 90° rotation. Fixing...")
            return img.rotate(-90, expand=True) # Rotate counter-clockwise to fix
        elif angle_str == "180":
            print("-> Detected upside-down image. Fixing...")
            return img.rotate(180, expand=True)
        elif angle_str == "270":
            print("-> Detected 270° rotation. Fixing...")
            return img.rotate(90, expand=True)
        else:
            print("-> Image is upright.")
            return img

    except Exception as e:
        print(f"Orientation check failed, proceeding with original: {e}")
        return img

def transcribe_exam_paper(uploaded_file, api_key:str):
    print("Processing exam image payload...")
# ///
    try:
        # Open and apply standard EXIF fixes first
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Initialize model
        model = get_vision_model(api_key)
        
        # AI Auto-Rotation 
        img = auto_rotate_image(img, model)
        
        # Now compress the FIXED image for the main transcription
        img.thumbnail((1600, 1600))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

    print('Extracting handwritten text...')
    
    # We no longer need to warn the model about rotation because we already fixed it!
    prompt_text = """
    You are an expert handwriting transcription engine. 
    Read this student's exam paper and extract all the text exactly as written. 
    Do not format it into JSON. Do not add commentary. Just give me the raw text of the answers.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ]
    )

    try:
        response = model.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None