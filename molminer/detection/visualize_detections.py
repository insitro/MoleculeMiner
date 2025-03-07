# Changed for core
"""
Author: Abhisek Dey

This Script Generates the Image-Based Overlay of the Detection Boxes on top of the patents


Assumes that local_visualize.py has already been run and the CSV files containing
detections have already been generated

OUTPUT: The output of the script is the the PDF Images organized by their names and
overlaid by their detected molecule diagrams. Default Output Location:
datasets/data/Test_Patents/processed/detection_overlaid/PDF1;PDF2;
..../Page_00001.png;Page_00002.png....

INPUT:
1. "in_dir": Folder Location Containing Processed PDF Images and the detection
                folder containing the CSVs
                (Default: datasets/data/Test_Patents/processed)

2. "out_dir": Folder where to save the box overlaid images
                (Default: datasets/data/Test_Patents/processed/overlaid)
"""

import argparse
import csv
import os
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm import tqdm


# Creates a dictionary containing all the images for every PDF found in the data directory provided
def read_all_images(root_dir) -> Dict[str, Dict[int, np.ndarray]]:
    all_pdfs = os.listdir(root_dir)
    all_pdfs = [pdf.split(".")[0] for pdf in all_pdfs if pdf.endswith(".pdf")]

    # Create the empty dictionary per pdf
    pdf_imgs: Dict[str, Dict[int, ndarray]] = {pdf: {} for pdf in all_pdfs}

    for pdf in tqdm(
        all_pdfs, total=len(all_pdfs), desc="Creating the Original PDF Images Dictionary"
    ):
        imgs = sorted(os.listdir(os.path.join(root_dir, pdf + ".pdf")))
        imgs_path = [os.path.join(root_dir, pdf + ".pdf", i) for i in imgs]
        for i, path in enumerate(imgs_path):
            img = cv2.imread(path, 1)
            pdf_imgs[pdf][int(imgs[i].split(".")[0].split("_")[1])] = img

    print("Finished Creating the Images dictionary")
    return pdf_imgs


# Create a dictionary from reading the boxes for each PDF CSV
def read_all_boxes(mode, root_dir) -> Dict[str, Dict[int, List]]:
    # Check mode and then read the appropriate directory
    if mode == "mol":
        folder = "detections"
    elif mode == "tab":
        folder = "detections_table"

    all_detections = os.listdir(os.path.join(root_dir, folder))
    all_detections_path = [os.path.join(root_dir, folder, pdf) for pdf in all_detections]

    # Create the empty dictionary per pdf
    pdf_boxes: Dict = {pdf.split(".")[0]: {} for pdf in all_detections}

    for i, det_csv in tqdm(
        enumerate(all_detections_path),
        total=len(all_detections_path),
        desc="Creating the PDF boxes dictionary",
    ):
        with open(det_csv, "r") as file:
            csvFile = csv.DictReader(file)
            # Append All Boxes Based on the PAge of the PDF
            for line in csvFile:
                if int(line["Page"]) not in pdf_boxes[all_detections[i].split(".")[0]].keys():
                    pdf_boxes[all_detections[i].split(".")[0]][int(line["Page"])] = [
                        [
                            float(line["x1"]),
                            float(line["y1"]),
                            float(line["x2"]),
                            float(line["y2"]),
                        ]
                    ]
                else:
                    pdf_boxes[all_detections[i].split(".")[0]][int(line["Page"])].append(
                        [
                            float(line["x1"]),
                            float(line["y1"]),
                            float(line["x2"]),
                            float(line["y2"]),
                        ]
                    )
        file.close()

    print("Finished Creating the PDF Boxes Dictionary")
    return pdf_boxes


# Script to write out the detected Images
def generate_overlay(images, boxes, out_dir):
    total_pages = 0

    # Collate boxes from the dictionaries and overlay them on the images
    pdf_list = images.keys()
    for pdf in tqdm(pdf_list, total=len(pdf_list), desc="Overlaying available boxes"):
        # Create the directory for storing the images
        pdf_dir = os.path.join(out_dir, pdf)
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)

        # Loop through the images and see whether there are diagrams for that page
        for page in images[pdf].keys():
            img = images[pdf][page]
            # If there are boxes found for that page, add the boxes for that page
            if page in boxes[pdf].keys():
                for box in boxes[pdf][page]:
                    start = (int(box[0]), int(box[1]))
                    end = (int(box[2]), int(box[3]))
                    color = (255, 0, 0)
                    thickness = 2
                    img = cv2.rectangle(img, start, end, color, thickness)

            # Write the current page
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(
                os.path.join(pdf_dir, f"Page_{page:0>{5}}.png"), bbox_inches="tight", dpi=100
            )
            plt.close()
            total_pages += 1

    print(f"Finished Overlaying Boxes. TOTAL PAGES OVERLAID = {total_pages}")


def overlay_process(args):
    pdf_images = read_all_images(args.in_dir)
    pdf_boxes = read_all_boxes(args.mode, args.in_dir)
    generate_overlay(images=pdf_images, boxes=pdf_boxes, out_dir=args.out_dir)


def parse_args() -> None:
    parser = argparse.ArgumentParser(description="ScanSSD: Scanning Single Shot MultiBox Detector")
    parser.add_argument(
        "--mode",
        default="mol",
        type=str,
        help="mol: Overlay Molecule Detections; tab: Overlay the Table Detections",
    )
    parser.add_argument(
        "--in_dir",
        default="datasets/data/Test_Patents/processed",
        type=str,
        help="Input Directory containing processed PDFs and Detected CSV files",
    )
    parser.add_argument(
        "--out_dir",
        default="datasets/data/Test_Patents/processed/overlaid",
        type=str,
        help="Output Directory to save the overlaid boxes",
    )

    args = parser.parse_args()

    # Start the Overlay Process
    overlay_process(args)


if __name__ == "__main__":
    parse_args()
