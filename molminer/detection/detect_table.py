# Changed for core
"""
Test Script for Table Detection and If it works for chemical patents
Trial 1: Using the Table Transformer for Object Detection
Trial 2: Using Layout Parser FasterRcnn trained on PubLayNet
"""

import csv
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

PDF_DIR = "datasets/data/Test_Patents/processed/"


def process_pdfs_tabtrans():
    pdf_list = os.listdir(PDF_DIR)
    pdf_list = [pdf for pdf in pdf_list if pdf.endswith(".pdf")]

    # Create the directory for storing the new CSVs
    csv_dir = os.path.join(PDF_DIR, "detections_table")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Initiate the Model and Image Processor
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    print("Processing all PDFs...")
    for i, pdf in enumerate(pdf_list):
        print(f"Processing PDF: {pdf}")
        # Get all the pages of the PDF:
        page_list = sorted(os.listdir(os.path.join(PDF_DIR, pdf)))
        page_list = [os.path.join(PDF_DIR, pdf, p) for p in page_list]

        # Create the empty lists to store the table boxes (per PDF)
        fields = ["Page", "x1", "y1", "x2", "y2"]
        page_boxes = []
        for j, page in tqdm(enumerate(page_list), total=len(page_list), desc="Pages Processed:"):
            page_img = Image.open(page).convert("RGB")

            inputs = image_processor(images=page_img, return_tensors="pt")
            outputs = model(**inputs)

            target_sizes = torch.tensor([page_img.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.75, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if label.item() == 0 and score.item() > 0.75:
                    box = [int(i) for i in box.tolist()]
                    page_boxes.append(
                        [f"{j + 1}", f"{box[0]}", f"{box[1]}", f"{box[2]}", f"{box[3]}"]
                    )

        # Now Write the page boxes for the PDF
        print(f"Writing CSV for PDF: {pdf}")
        filename = os.path.join(csv_dir, pdf.split(".")[0] + ".csv")
        with open(filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(page_boxes)

    print("Finished Processing ALL PDFs...")


# def process_pdfs_lp():
#     pdf_list = os.listdir(PDF_DIR)
#     pdf_list = [pdf for pdf in pdf_list if pdf.endswith(".pdf")]

#     # Create the directory for storing the new CSVs
#     csv_dir = os.path.join(PDF_DIR, "detections_table")
#     if not os.path.exists(csv_dir):
#         os.makedirs(csv_dir)

#     # Initialize the LayoutParser Model
#     model = lp.Detectron2LayoutModel(
#         config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",  # In model catalog
#         label_map={
#             0: "Text",
#             1: "Title",
#             2: "List",
#             3: "Table",
#             4: "Figure",
#         },  # In model`label_map`
#         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],  # Optional
#     )


if __name__ == "__main__":
    if sys.argv[1] == "tabtrans":
        process_pdfs_tabtrans()
    # elif sys.argv[1] == "lp":
    #     # process_pdfs_lp()
