import os
from pdf2image import convert_from_path
from google.cloud import vision
import io

def pdf_to_images(pdf_path, output_folder="images", poppler_path=r"C:\\poppler\\Library\\bin"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path, output_folder=output_folder, poppler_path=poppler_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

def ocr_with_google(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return "\n".join([text.description for text in texts])

def create_ground_truth_from_pdf(pdf_path, output_folder="ground_truth", poppler_path=r"C:\\poppler\\Library\\bin"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    image_paths = pdf_to_images(pdf_path, poppler_path=poppler_path)

    all_text = []
    for image_path in image_paths:
        ocr_text = ocr_with_google(image_path)
        all_text.append(ocr_text)

    ground_truth_path = os.path.join(output_folder, f"{base_name}.txt")
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"Ground truth created: {ground_truth_path}")

# Example usage
if __name__ == "__main__":
    input_pdf_folder = r"fwwchloosereportfaxexamples"
    for file in os.listdir(input_pdf_folder):
        if file.lower().endswith(".pdf"):
            create_ground_truth_from_pdf(os.path.join(input_pdf_folder, file))
