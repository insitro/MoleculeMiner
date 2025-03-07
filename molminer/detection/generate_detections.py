# Changed for core
import csv
import os
import sys
from glob import glob

import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

MAX_PAGES_WIDTH = (
    5  # Max number of page number digits possible per pdf; 3 - 999, 4 - 9999, 5 - 99999
)


# Helper function to read all PDFs and store their corresponding image pages
def read_pdfs(read_path, write_path):
    # Check if Image Saving Location Exists, if not make the directory
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Check Write Path to see if there are requisite number of folders (1 Per PDF)
    num_pdfs = len(glob(os.path.join(read_path, "*.pdf")))
    num_folders = len(os.listdir(write_path))
    if num_folders >= num_pdfs:
        print("SKIPPING generating images as folders already found")
        return

    # Else: Start Generating Images
    patent_cls = os.listdir(read_path)
    patent_cls = [os.path.join(read_path, dir) for dir in patent_cls]
    pdfs = []
    pdf_ps = []
    for dir in patent_cls:
        all_cls_files = os.listdir(dir)
        all_cls_pdfs = [f for f in all_cls_files if f.endswith(".pdf")]
        all_cls_pdfs_path = [os.path.join(dir, f) for f in all_cls_pdfs]
        pdfs.extend(all_cls_pdfs)
        pdf_ps.extend(all_cls_pdfs_path)

    pdf_paths = {pdf_name: pdf_path for pdf_name, pdf_path in zip(pdfs, pdf_ps)}

    # Convert the PDFs to Images
    print("Converting ALL PDFs to IMAGES using a DPI of 300")
    for name in tqdm(pdf_paths.keys(), total=len(pdf_paths.keys())):
        images = convert_from_path(pdf_path=pdf_paths[name], dpi=300)

        # Save all the images to disk
        save_path = os.path.join(write_path, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, img in enumerate(images):
            img.save(os.path.join(save_path, f"Page_{i + 1:0>{MAX_PAGES_WIDTH}}.png"), "png")

    print(f"FINISHED Converting all PDFs to Images, available at {write_path}....")


# Helper function to generate the CSV file per PDF of the detections
def generate_detections(save_location, conf: float = 0.20, iou_thresh: float = 0.4):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Creating Directory for PDF Detection CSVs...")
    root_csv_dir = os.path.join(save_location, "detections")
    if not os.path.exists(root_csv_dir):
        os.makedirs(root_csv_dir, exist_ok=True)

    print("Loading the best YOLO weights...")
    model = YOLO("yolov8n.pt")

    print("Compiling List of PDFs and their images")
    pdfs = os.listdir(save_location)
    pdfs = [p for p in pdfs if p.endswith(".pdf")]  # Only Read PDF Image Directories

    for pdf in pdfs:
        print(f"Detection Overview for PDF: {pdf}")
        pdf_imgs_paths = sorted(os.listdir(os.path.join(save_location, pdf)))
        pdf_imgs = []
        pdf_sizes = []
        for page in pdf_imgs_paths:
            img = Image.open(os.path.join(save_location, pdf, page))
            np_img = np.array(img)
            pdf_imgs.append(np_img)
            pdf_sizes.append((img.height, img.width))

        results = model.predict(pdf_imgs, conf=conf, iou=iou_thresh, device=device)

        # Create a New CSV File for the PDF
        filename = os.path.join(root_csv_dir, pdf.rstrip(".pdf") + ".csv")
        fields = ["Page", "x1", "y1", "x2", "y2"]
        page_boxes = []
        for i, img_results in enumerate(results):
            boxes = img_results.boxes
            for box in boxes:
                if len(box.xyxy):
                    b = box.xyxy.cpu().numpy()[0].astype(np.float32)
                    b = [f"{i + 1}", f"{b[0]:.2f}", f"{b[1]:.2f}", f"{b[2]:.2f}", f"{b[3]:.2f}"]
                    page_boxes.append(b)

        # Write the PDF CSV
        print(f"Writing CSV for PDF: {pdf}")
        with open(filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(page_boxes)


if __name__ == "__main__":
    data_location = sys.argv[1]  # Location of Patents Directory
    save_location = sys.argv[2]  # Save Location for Images, Detection CSVs etc.

    # Generate Images out of the Images
    read_pdfs(data_location, save_location)
    # Generate CSV Files containing detections for each PDF (Location: 'save_location'/detections)
    generate_detections(save_location)
