import os
import time
from OCR_Models.BaseOCR import BaseOCR
# from OCR_Models.DocTROCR import DocTROCR
from OCR_Models.SuryaOCR import SuryaOCR
from OCR_Models.TesseractOCRTool import TesseractOCR
#from OCR_Models.PaddleOCR import PaddleOCR
from OCR_Models.EasyOCR import EasyOCR

# Directory containing your PDF/Image files
directory_path = r"fwwchloosereportfaxexamples"
output_folder = os.path.join(directory_path, "OCR_results")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Instantiate your OCR models
ocr_models = {
     'TesseractOCR': TesseractOCR(language="eng"),
     'SuryaOCR': SuryaOCR(language="en"),
    # 'DocTROCR': DocTROCR(language="en"),
    # 'PaddleOCR': PaddleOCR(language="en")
    'EasyOCR': EasyOCR(language="en")
}

summary_file = os.path.join(output_folder, "OCR_summary.txt")
with open(summary_file, "w", encoding="utf-8") as summary:
    summary.write("Filename,Model,Processing Time (seconds),Average Confidence\n")

    # Run all OCR models on each document
    for file in os.listdir(directory_path):
        if file.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory_path, file)
            print(f"Processing: {file_path}")

            for model_name, model_instance in ocr_models.items():
                start_time = time.time()
                text_output = model_instance.ocr_image(file_path)
                conf_output = model_instance.ocr_image_with_conf(file_path)
                elapsed_time = time.time()
                end_time = elapsed_time - start_time

                base_filename = os.path.splitext(file)[0]

                # Save plain OCR text
                text_file = os.path.join(output_folder, f"{base_filename}_{model_name}_OCR.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                print(f"{model_name} results saved for {file}")
         
                # Save OCR text with confidence
                conf_file = os.path.join(output_folder, f"{base_filename}_{model_name}_OCR_with_conf.txt")
                confidences = []
                with open(conf_file, "w", encoding="utf-8") as f:
                    for entry in conf_output:
                        confidences.append(entry['confidence'])
                        f.write(f"{entry['text']} (Confidence: {entry['confidence']})\n")

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                summary.write(f"{file},{model_name},{end_time:.2f},{avg_confidence:.2f}\n")

                print(f"{model_name} results saved for {file} with confidence and timing")

