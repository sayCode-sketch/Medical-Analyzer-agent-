import streamlit as st
import pytesseract
from PIL import Image
import pdfplumber
import re
import os
import shutil
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TESSERACT_PATH = shutil.which("tesseract")
if not TESSERACT_PATH:
    st.error("⚠️ Tesseract OCR is not installed. Please install it:\n"
             "```bash\nbrew install tesseract\n```")
else:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(file):
    """Extracts text from PDF or image file."""
    file_name = file.name.lower()

    try:
        if file_name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        else:
            image = Image.open(file).convert("RGB")
            return pytesseract.image_to_string(image)
    except Exception as e:
        return f"ERROR: {str(e)}"

def parse_lab_values(text):
    """
    Tries to extract lab values in the format:
    TestName: number unit
    Returns a dict, or empty if no matches found.
    """
    pattern = r"([A-Za-z\s]+):?\s*([\d.]+)\s*(\w+)?"
    matches = re.findall(pattern, text)
    return {name.strip(): f"{value} {unit or ''}".strip()
            for name, value, unit in matches} if matches else {}

def interpret_with_gpt(text, parsed_data):
    """
    Generates a detailed, patient-friendly summary
    whether or not numeric lab values are found.
    """
    prompt = (
        "You are a compassionate medical assistant. "
        "You will be given the text of a medical report, along with any structured lab values found. "
        "Your task is to create a detailed, patient-friendly summary that is warm, clear, and reassuring. "
        "Follow this structure:\n\n"
        "1. **Visit Details** - Mention patient's name, date of visit, and doctor's name & specialization if available.\n"
        "2. **Examination Summary** - Describe what was checked, any tests, and key findings.\n"
        "3. **Diagnosis** - Clearly explain the health status.\n"
        "4. **Treatment / Recommendations** - If no medication is needed, explain why and give healthy living advice.\n"
        "5. **Overall Conclusion** - End with a reassuring statement.\n\n"
        "Avoid medical jargon and keep it friendly and easy to understand.\n\n"
    )

    if parsed_data:
        prompt += "Structured Lab Values:\n"
        for test, value in parsed_data.items():
            prompt += f"- {test}: {value}\n"
        prompt += "\n"

    prompt += "Full Extracted Report Text:\n" + text.strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

st.set_page_config(page_title="🫁 Medical Report Analyzer")

st.title("🫁 Medical Report Analyzer")
st.write("Upload a medical report (PDF or Image). This app extracts text, finds lab values, and generates a **detailed, patient-friendly summary** using GPT.")

uploaded_file = st.file_uploader("Upload your report", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
       
        if not TESSERACT_PATH and uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            st.error("Image OCR requires Tesseract OCR. Please install it first.")
        else:
            st.info("Extracting text...")
            raw_text = extract_text(uploaded_file)

            if not raw_text.strip() or raw_text.startswith("ERROR:"):
                st.error("No text found or error during extraction.")
            else:
               
                st.subheader("📜 Extracted Text")
                st.text(raw_text)

                parsed_data = parse_lab_values(raw_text)
                if parsed_data:
                    st.subheader("📊 Parsed Lab Values")
                    st.json(parsed_data)
                else:
                    st.info("No structured lab values detected. Summary will be based on full text.")

                st.info("Analyzing results with GPT...")
                summary = interpret_with_gpt(raw_text, parsed_data)
                st.subheader("📝 Detailed Summary")
                st.write(summary)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
