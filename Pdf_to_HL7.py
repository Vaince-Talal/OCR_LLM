import os

import requests
from pdf2image import convert_from_path
from hl7apy.core import Message, Segment
from gpt4all import GPT4All
import easyocr  # Using EasyOCR as the OCR tool


# Function to convert PDF to image
def pdf_to_image(pdf_path, output_folder="images", poppler_path=r"C:\poppler\Library\bin"):
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
def ocr_image_with_structure(image_path):
    """
    Perform OCR and attempt to maintain text structure using bounding boxes.
    """
    reader = easyocr.Reader(["en"], gpu=True)
    result = reader.readtext(image_path, detail=1)  # Get detailed results with bounding boxes

    structured_text = []
    for detection in result:
        bbox, text, confidence = detection
        # Add bounding box coordinates or other structural markers if needed
        structured_text.append({"bbox": bbox, "text": text, "confidence": confidence})

    # Sort by vertical (y-coordinate) position to maintain reading order
    structured_text.sort(key=lambda x: x["bbox"][0][1])  # Sort by the top-left y-coordinate

    # Join text lines while maintaining structure
    result = "\n".join(item["text"] for item in structured_text)
    print(result)
    return result



# Function to extract patient information using LLM SDK
# def extract_patient_info_with_llm(ocr_text):
#     llm = GPT4All(model_name="qwen2.5-coder-7b-instruct-q4_0")
#     prompt = f"Extract patient information from the following text: \n{ocr_text}\nPlease provide Name, Date of Birth, ID, and Address in JSON format. Date of Birth will usually be denoted by DOB in the docuemnet. ID is the MRN or can be potienally something else. "
#     with llm.chat_session():
#         response = llm.generate(prompt, max_tokens=1024)
#     try:
#         print(response)
#         patient_info = eval(response)  # Assuming the LLM provides valid JSON-like output
#     except Exception as e:
#         print("Error parsing LLM response:", e)
#         patient_info = {
#             "Name": "Unknown",
#             "DOB": "Unknown",
#             "ID": "Unknown",
#             "Address": "Unknown"
#         }
#     return patient_info
def extract_patient_info_with_api(ocr_text):
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
                    '    "address": "string" - should be in #num street, city, province,\n'
                    '    "mrn": "string (numbers only)",\n'
                    '    "category": "string"'                                
                    "}\n\n"
                    "If any information is missing or cannot be reasonably inferred, return 'Unknown' for that field. "
                    "Make reasonable assumptions for messy data, and ensure DOB reflects context (e.g., adjust based on age if provided)."
                )
            }
        ],
        "max_tokens": 512,  # Adjust as needed for expected response size
        "temperature": 0.4  # Consistent results with slight variability for inferences
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            # Extract the assistant's response
            content = result['choices'][0]['message']['content']
            print(content)
            return eval(content)  # Convert the response into a dictionary
        else:
            print(f"Error from GPT4All server: {response.status_code}")
            print(response.text)
    except Exception as e:
        print("Error connecting to GPT4All server:", e)

    # Return a default response if something goes wrong
    return {
        "name": "Unknown",
        "dob": "Unknown",
        "id": "Unknown",
        "address": "Unknown"
    }

# Function to generate an HL7 message
def generate_hl7_message(patient_info):
    msg = Message("ADT_A01")

    # Populate the MSH segment
    msg.msh.msh_1 = "|"
    msg.msh.msh_2 = "^~\&"
    msg.msh.msh_3 = "SendingApplication"
    msg.msh.msh_4 = "SendingFacility"
    msg.msh.msh_5 = "ReceivingApplication"
    msg.msh.msh_6 = "ReceivingFacility"
    msg.msh.msh_7 = "20241227120000"
    msg.msh.msh_9 = "ADT^A01"
    msg.msh.msh_10 = "12345"
    msg.msh.msh_11 = "P"

    # Populate the PID segment
    try:
        pid = Segment("PID")
        pid.pid_3 = patient_info['ID']
        pid.pid_5 = patient_info['Name']
        pid.pid_7 = patient_info['Date of Birth']
        pid.pid_11 = patient_info['Address']
        msg.add(pid)
    except:
        print("Something went wrong in translation")


    return msg.to_er7()


# Main script
def main(directory_path):
    # Step 1: Process all PDFs in the given directory
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        print(f"Processing: {pdf_path}")

        # Convert PDF to images
        images = pdf_to_image(pdf_path)

        # Step 2: Perform OCR and extract patient info
        all_patient_info = []
        for image_path in images:
            ocr_text = ocr_image_with_structure(image_path)
            patient_info = extract_patient_info_with_api(ocr_text)
            all_patient_info.append(patient_info)

        # Step 3: Generate HL7 messages
        hl7_messages = []
        for patient_info in all_patient_info:
            hl7_message = generate_hl7_message(patient_info)
            hl7_messages.append(hl7_message)

        # Print or save HL7 messages
        for idx, hl7_message in enumerate(hl7_messages):
            print(f"\nHL7 Message for {pdf_file} - Page {idx + 1}:")
            print(hl7_message)


if __name__ == "__main__":
    directory_path = r"C:\Users\talal\PycharmProjects\OCR_LLM\fwwchloosereportfaxexamples"  # Replace with your directory path containing PDFs
    main(directory_path)
