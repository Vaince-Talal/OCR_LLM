import os
import pandas as pd
import difflib

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def ordered_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

def unordered_similarity(text1, text2):
    words1 = sorted(text1.lower().split())
    words2 = sorted(text2.lower().split())
    return difflib.SequenceMatcher(None, " ".join(words1), " ".join(words2)).ratio() * 100

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) * 100 if union else 100

# Folders
ground_truth_folder = "ground_truth"
ocr_results_folder = "fwwchloosereportfaxexamples/OCR_results"

results = []

# Loop through all ground truth text files
for gt_file in os.listdir(ground_truth_folder):
    if gt_file.endswith(".txt"):
        doc_name = os.path.splitext(gt_file)[0]
        gt_path = os.path.join(ground_truth_folder, gt_file)
        ground_text = load_text(gt_path)

        # Match OCR outputs for this document
        for file in os.listdir(ocr_results_folder):
            if (
                file.endswith("_OCR.txt") and
                not file.endswith("_OCR_with_conf.txt") and
                doc_name in file
            ):
                model_name = file.split("_")[-2]
                ocr_path = os.path.join(ocr_results_folder, file)
                ocr_text = load_text(ocr_path)

                results.append({
                    "Document": doc_name,
                    "Model": model_name,
                    "Ordered Similarity (%)": round(ordered_similarity(ground_text, ocr_text), 2),
                    "Unordered Similarity (%)": round(unordered_similarity(ground_text, ocr_text), 2),
                    "Jaccard Similarity (%)": round(jaccard_similarity(ground_text, ocr_text), 2)
                })

# Save to CSV
df = pd.DataFrame(results)
df.sort_values(by=["Document", "Ordered Similarity (%)"], ascending=False, inplace=True)
df.to_csv("ocr_model_comparison.csv", index=False)
