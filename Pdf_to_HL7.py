import os
import requests
from pdf2image import convert_from_path
from hl7apy.core import Message, Segment
from gpt4all import GPT4All
import easyocr  # Using EasyOCR as the OCR tool
from google.cloud import vision
import io

# Function to convert PDF to image
def pdf_to_image(pdf_path, output_folder="images", poppler_path=r"C:\\poppler\\Library\\bin"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, output_folder=output_folder, poppler_path=poppler_path)
    image_paths = []

    for i, image in enumerate(images):
        output_image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(output_image_path, "PNG")
        image_paths.append(output_image_path)

    return image_paths

# Function to perform OCR on an image using EasyOCR
def ocr_image_easyocr(image_path, output_file):
    reader = easyocr.Reader(["en"], gpu=True)
    result = reader.readtext(image_path, detail=0)  # Get text only

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result))

    return "\n".join(result)

# Function to perform OCR on an image using Google Vision API
def ocr_image_google(image_path, output_file):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    extracted_text = "\n".join([text.description for text in texts])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    return extracted_text

# Function to extract patient information using GPT4All and save the response
def extract_patient_info_with_api(ocr_text, output_file):
    url = "http://localhost:4891/v1/chat/completions"  # GPT4All server endpoint
    payload = {
        "model": "Reasoner v1",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Extract patient information from the following text: \n{ocr_text}\n"
                    "Please provide the following details in JSON format:\n"
                    "1. Name\n"
                    "2. Date of Birth (DOB) in YYYY-MM-DD format.\n"
                    "3. ID\n"
                    "4. Address - The address should only be \n"
                    "5. MRN (Medical Record Number). If the MRN contains letters, correct it to an appropriate numeric-only value.\n\n"
                    "6. The category of the document. Should be one or two words"
                    "Ensure the output JSON is formatted as:\n"
                    "{\n"
                    '    "name": "string",\n'
                    '    "dob": "string in YYYY-MM-DD",\n'
                    '    "id": "string",\n'
                    '    "address": "string",\n'
                    '    "mrn": "string (numbers only)",\n'
                    '    "category": "string"'
                    "}\n\n"
                    "If any information is missing or cannot be reasonably inferred, return 'Unknown' for that field. "
                    "Make reasonable assumptions for messy data, and ensure DOB reflects context (e.g., adjust based on age if provided)."
                )
            }
        ],
        "max_tokens": 512,
        "temperature": 0.4
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            return eval(content)
        else:
            print(f"Error from GPT4All server: {response.status_code}")
            print(response.text)
    except Exception as e:
        print("Error connecting to GPT4All server:", e)

    return {
        "name": "Unknown",
        "dob": "Unknown",
        "id": "Unknown",
        "address": "Unknown",
        "mrn": "Unknown",
        "category": "Unknown"
    }

# Main script
def main(directory_path):
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    output_folder = os.path.join(directory_path, "OCR_results")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"Processing: {pdf_path}")

        images = pdf_to_image(pdf_path)

        for idx, image_path in enumerate(images):
            easyocr_output_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page_{idx + 1}_OCR_EasyOCR.txt")
            googleocr_output_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page_{idx + 1}_OCR_Google.txt")
            easyocr_llm_response_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page_{idx + 1}_EasyOCR_LLM_Response.txt")
            googleocr_llm_response_file = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page_{idx + 1}_GoogleOCR_LLM_Response.txt")

            easyocr_text = ocr_image_easyocr(image_path, easyocr_output_file)
            google_ocr_text = ocr_image_google(image_path, googleocr_output_file)

            extract_patient_info_with_api(easyocr_text, easyocr_llm_response_file)
            extract_patient_info_with_api(google_ocr_text, googleocr_llm_response_file)

            print(f"Patient Info for {pdf_file} - Page {idx + 1} saved.")

if __name__ == "__main__":
    directory_path = r"C:\Users\talal\PycharmProjects\OCR_LLM\fwwchloosereportfaxexamples"
    main(directory_path)
