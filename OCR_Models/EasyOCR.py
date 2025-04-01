from easyocr import Reader
from OCR_Models.BaseOCR import BaseOCR

class EasyOCR(BaseOCR):
    def __init__(self, language="en", gpu=False, poppler_path=r"C:\\poppler\\Library\\bin"):
        super().__init__(language, gpu, poppler_path)
     
        self.reader = Reader(lang_list=[language], gpu=gpu)

    def ocr_image(self, file_path):
        # Convert PDF to images (assuming convert_pdf_to_img is implemented in BaseOCR)
        images = self.convert_pdf_to_img(file_path)
        text_results = []
        
        for img_path in images:
            # Read text from the image
            result = self.reader.readtext(img_path)
            
            text = '\n'.join([line[1] for line in result])  # Join the text elements with newlines
            text_results.append(text)
        

        return '\n\n'.join(text_results)

    def ocr_image_with_conf(self, file_path):

        images = self.convert_pdf_to_img(file_path)
        detailed_results = []

        for img_path in images:
            result = self.reader.readtext(img_path)
            for line in result:
                detailed_results.append({"text": line[1], "confidence": line[2]*100})
        
        return detailed_results
