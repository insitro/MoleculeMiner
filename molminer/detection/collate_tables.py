# Changed for core
"""
Author - Abhisek Dey
MANUAL Collation of Data for Chosen Patents

Manually describing START-END pages for Patents and their individual quirks
"""

import csv
import os
from collections import defaultdict
from functools import partial
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from shapely import geometry
from tqdm import tqdm

# Save Directory for storing the Image-SMILES Combo
SAVE_DIR = "MoleculeBank"
MAX_PAGES_WIDTH = 5


# Read the specific excel file and only keep columns of interest
def read_excel(excl_path, sheet_name, columns, drop_cids, duplicates) -> Dict[int, str]:
    data_frame = pd.read_excel(excl_path, sheet_name)
    data_frame = data_frame[columns]

    # Canonicalize the SMILES
    data_frame[columns[-1]] = data_frame[columns[-1]].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x))
    )

    # cid2smiles compilation
    cid2smiles = data_frame.set_index(columns[0])[columns[-1]].to_dict()

    # Get actual cids to drop (Doing this way as some cid fields have more than one cid
    # In some patents)
    if len(drop_cids):
        remove_cids = []
        for cid in cid2smiles.keys():
            if cid in drop_cids:
                remove_cids.append(cid)
        for r_id in remove_cids:
            cid2smiles.pop(r_id, None)

    # If duplicates in CID rows make them separate
    if duplicates:
        temp_dict = {}
        for cid in cid2smiles.keys():
            smi = cid2smiles[cid]
            if ";" not in cid:
                act_id = int(cid.split("_")[-1])
                temp_dict[act_id] = smi
            else:  # Split by number of compound IDs having same SMILES
                cids = cid.split(";")
                for idx in cids:
                    int_id = int(idx.split("_")[-1])
                    temp_dict[int_id] = smi

        # Now re-write the cid2smiles dictionary maintaining CID order
        cid2smiles = {}
        cids = sorted(temp_dict.keys())
        for cid in cids:
            cid2smiles[cid] = temp_dict[cid]

    return cid2smiles


# Sort by estimating writing lines first and then sorting each from left to right
def sort_special(page_boxes):  # -> Dict[int, List]:
    line2box = defaultdict(list)

    # First sort the boxes just along y-axis
    y_sort_idx = np.argsort(page_boxes[:, 1])
    page_boxes = page_boxes[y_sort_idx]

    cur_line = 1
    # Add the first box to the first writing line and keep a track of it
    line2box[cur_line].append(page_boxes[0])
    added_lines = [0]

    # Special sorting algorithm
    i = 0
    while i < len(page_boxes) - 1:
        for j in range(i + 1, len(page_boxes)):
            rectA = page_boxes[i]
            rectB = page_boxes[j]

            # Check for rectA if it is already added or not
            if i not in added_lines:
                line2box[cur_line].append(rectA)
                added_lines.append(i)

            # Check for the condition of same writing line if rectB not already added
            if j not in added_lines:
                rectA_y = rectA[1]
                rectB_y = rectB[1]
                rectA_h = rectA[3] - rectA[1]
                # Add to next writing line if difference greater than height
                if (rectB_y - rectA_y) < 0:
                    raise Exception("Sorting Error. Found 2nd box y less than 1st box y")
                if rectB_y - rectA_y > rectA_h:
                    cur_line += 1
                    line2box[cur_line].append(rectB)
                    added_lines.append(j)
                    break
                else:
                    line2box[cur_line].append(rectB)
                    added_lines.append(j)
        i = j + 1

    # Now sort all the boxes per line along the x-direction
    for line in line2box.keys():
        line_boxes = np.array(line2box[line]).reshape((-1, 4))
        # Sort Along X-Direction
        x_sort_idxs = np.argsort(line_boxes[:, 0])
        line_boxes = line_boxes[x_sort_idxs]
        line2box[line] = line_boxes

    return line2box


def sort_ordinary(page_boxes) -> Dict[int, ndarray]:
    # Just for restructuring into one line for downstream processing
    line2box: defaultdict = defaultdict(partial(ndarray, 0))
    temp_boxes = []
    for box in page_boxes:
        temp_boxes.append(box)

    # Convert all boxes to a numpy array
    line2box[1] = np.array(temp_boxes).reshape((-1, 4))
    return line2box


# Read the specific Detection CSV and filter by pages and/or specific
# smallest page box
def read_boxes(csv_path, start, end, offset, adds, to_remove, sort_by_line=True):
    # Create an empty defaultdict to store pages and their boxes
    pg2boxes = defaultdict(list)
    total_valid = 0

    with open(csv_path, "r") as file:
        csvFile = csv.DictReader(file)
        # Read all the boxes of all the pages first
        for line in csvFile:
            page = int(line["Page"])
            pg2boxes[page].append(
                [float(line["x1"]), float(line["y1"]), float(line["x2"]), float(line["y2"])]
            )
    file.close()

    # Only use the valid pages for processing now
    fil_pg2boxes = {}
    for page in range(start, end + 1):
        page_boxes = np.array(pg2boxes[page]).reshape((-1, 4))
        # Sort top-down
        y_sort_idx = np.argsort(page_boxes[:, 1])
        page_boxes = page_boxes[y_sort_idx]

        # If Offset, remove the top off boxes:
        if page == start and offset:
            page_boxes = page_boxes[offset:]
        # If any small boxes to remove in page
        if page in adds:
            num_remove = to_remove[page]
            # Find the area of all boxes and sort them by area
            areas = []
            for box in page_boxes:
                poly = geometry.box(box[0], box[1], box[2], box[3])
                areas.append(poly.area)
            area_sort_idxs = np.argsort(areas)
            page_boxes = page_boxes[area_sort_idxs]
            # Remove the number of small boxes
            page_boxes = page_boxes[num_remove:]

        # Sort special (approximating writing lines)
        if sort_by_line:
            line2box = sort_special(page_boxes)
        else:
            line2box = sort_ordinary(page_boxes)
        fil_pg2boxes[page] = line2box
        total_valid += len(page_boxes)

    return fil_pg2boxes


"""
Actual Function to Generate Molecule Images from the Collated CIDs and Detected Boxes

ARGS:
1. imgs_path: Path to the directory containing the page images of the PDF
2. cid2smiles: Refined dictionary containing the relevant CIDS to
3. pg2boxes: Containing a dioctinary of pages which contains a dictionary of writing
                lines and their boxes sorted along the writing direction

RETURNS: None

Saves the molecul images defined by the boxes according to their CID and generates a CSV file
with the info: CID-SMILES
"""


def create_molecule_bank(imgs_path, cid2smiles, pg2boxes, box_offsets):
    # Get the PDF Name from Images Path and create its directory
    pdf = imgs_path.split("/")[-1]
    pdf_save_path = os.path.join(SAVE_DIR, pdf)
    pdf_rdkit_save_path = os.path.join(SAVE_DIR, pdf + "_rdkit")
    if not os.path.exists(pdf_save_path):
        os.makedirs(pdf_save_path)
    if not os.path.exists(pdf_rdkit_save_path):
        os.makedirs(pdf_rdkit_save_path)

    # Create a directory for storing the CSVs if it does not already exist
    csv_dir = os.path.join(SAVE_DIR, "csvs")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # List all the pages where relevant molecules are
    pages = pg2boxes.keys()

    # List all the CIDs
    cids = sorted(cid2smiles.keys())
    cur_iteration = 0

    for page in tqdm(sorted(pages), total=len(pages), desc="Processing Pages..."):
        page_img_fname = f"Page_{page:0>{MAX_PAGES_WIDTH}}.png"
        img_path = os.path.join(imgs_path, page_img_fname)
        img = cv2.imread(img_path, 1)
        for line in sorted(pg2boxes[page].keys()):
            for box in pg2boxes[page][line]:
                # Add offsets
                act_box = [
                    int(box[0] - box_offsets[0]),
                    int(box[1] - box_offsets[1]),
                    int(box[2] + box_offsets[2]),
                    int(box[3] + box_offsets[3]),
                ]
                # Cut out that box containing molecule
                mol_img = img[act_box[1] : act_box[3], act_box[0] : act_box[2], :]
                mol_img = Image.fromarray(mol_img)
                cid = cids[cur_iteration]
                smi = cid2smiles[cid]
                # Save the molecule image
                mol_img.save(os.path.join(pdf_save_path, f"CID_{cid:0>{MAX_PAGES_WIDTH}}.png"))
                # Render the RDKit equivalent of the SMILES
                rd_mol = Chem.MolFromSmiles(smi)
                Draw.MolToFile(
                    rd_mol, os.path.join(pdf_rdkit_save_path, f"CID_{cid:0>{MAX_PAGES_WIDTH}}.png")
                )
                cur_iteration += 1

    # Now save the csv for the CID2SMILES
    fields = ["CID", "SMILES"]
    with open(os.path.join(csv_dir, pdf.split(".")[0] + ".csv"), "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for cid in sorted(cid2smiles.keys()):
            csvwriter.writerow([cid, cid2smiles[cid]])
    csvfile.close()


def process_W02017189823A2():
    imgs_path = "datasets/data/Test_Patents/processed/WO2017189823A2.pdf"
    mol_det_csv = "datasets/data/Test_Patents/processed/detections/WO2017189823A2.csv"
    excl_path = "datasets/data/Test_Patents/WO2017189823A2.xlsx"
    box_offsets = [10, 5, 30, 10]

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Read the excel sheet for only sheet and columns of interest
    sheet_name = "Sheet0"
    columns = ["iD", "smiles"]
    drop_cids = [10, 131, 218, 631, 632, 721, 722, 725, 754, 755, 1093]
    duplicates = False
    cid2smiles = read_excel(excl_path, sheet_name, columns, drop_cids, duplicates)

    # Read the boxes containing only the specific pages with tables and sort them
    start = 70
    end = 99
    offset = 2
    adds = []
    to_remove = {}
    pg2boxes = read_boxes(mol_det_csv, start, end, offset, adds, to_remove)

    # Create the Images and the CID-SMILES csv
    create_molecule_bank(imgs_path, cid2smiles, pg2boxes, box_offsets)


def process_WO2013086208A1():
    imgs_path = "datasets/data/Test_Patents/processed/WO2013086208A1.pdf"
    mol_det_csv = "datasets/data/Test_Patents/processed/detections/WO2013086208A1.csv"
    excl_path = "datasets/data/Test_Patents/WO2013086208A1.xlsx"
    box_offsets = [10, 5, 30, 10]

    # Create the save directory if it does not exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Read the excel sheet for only sheet and columns of interest
    sheet_name = "Sheet1"
    columns = ["COMPOUND_ID", "SMILES"]
    drop_cids = []
    duplicates = True
    cid2smiles = read_excel(excl_path, sheet_name, columns, drop_cids, duplicates)

    # Read the boxes containing only the specific pages with tables and sort them
    start = 14
    end = 126
    offset = 0
    adds = [39, 42, 45]
    to_remove = {39: 1, 42: 1, 45: 1}
    sort_by_line = False
    pg2boxes = read_boxes(mol_det_csv, start, end, offset, adds, to_remove, sort_by_line)

    # Create the Images and the CID-SMILES csv
    create_molecule_bank(imgs_path, cid2smiles, pg2boxes, box_offsets)


if __name__ == "__main__":
    process_W02017189823A2()
    process_WO2013086208A1()
