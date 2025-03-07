"""
*********************PIPELINE PROCESSING UTILS *********************
AUTHOR: Abhisek Dey (Summer Intern, 2024)

Contains all the utils for:
1. Conversion of PDFs to Images at 300 DPI
2. Detection of Molecule Regions from PDF Images
3. Converting the individual molecule images to Canonical SMILES strings with confidence scores
"""

import argparse
import copy
import logging
import math
import multiprocessing
import os
from typing import List, Tuple

import cv2
import numpy as np
import pymupdf as pypdf
import torch
from PIL import Image
from rdkit import Chem
from SmilesPE.pretokenizer import atomwise_tokenizer
from tqdm import tqdm
from ultralytics import YOLO

from molminer.pipeline.data_structures import PDF, Page
from molminer.pipeline.ocr import add_box_and_text_over_image, detect_text_orientation
from MolScribe.molscribe.interface import MolScribe

logger = logging.getLogger("molminer.pipeline.utils")


def canonicalize_smiles(
    smiles: str,
    ignore_chiral: bool = False,
    ignore_cistrans: bool = False,
    replace_rgroup: bool = True,
) -> Tuple[str, bool]:
    if not isinstance(smiles, str) or smiles == "":
        return "", False
    if ignore_cistrans:
        smiles = smiles.replace("/", "").replace("\\", "")
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == "[" and token[-1] == "]":
                symbol = token[1:-1]
                if symbol[0] == "R" and symbol[1:].isdigit():
                    tokens[j] = f"[{symbol[1:]}*]"
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = "*"
        smiles = "".join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except Exception:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def pdf_to_img(pdf_path: str, start: int, stop: int) -> List[np.ndarray]:
    """Converts A Single PDF to a List of Numpy Images"""
    doc = pypdf.open(pdf_path)
    page_imgs = []
    # PIL Image to Numpy Image
    for page in doc[start:stop]:
        page_pix = page.get_pixmap(dpi=300)
        page_img = Image.frombytes("RGB", (page_pix.width, page_pix.height), page_pix.samples)
        # Convert PIL to Numpy
        np_page_img = np.array(page_img, dtype=np.uint8)
        page_imgs.append(np_page_img)
    return page_imgs


def convert_pdfs(args: argparse.Namespace) -> List[PDF]:
    """Converts a List/Directory of PDFs into Images"""
    # First check if the input path is a PDF or a directory
    pdf_paths = []
    in_path = args.in_pdfs
    if not os.path.isdir(in_path):
        pdf_paths.append(in_path)
    else:
        pdfs = os.listdir(in_path)
        pdf_paths = [os.path.join(in_path, pdf) for pdf in pdfs if pdf.endswith(".pdf")]

    # Setup multiprocessing for converting PDF to page images
    cpu_cnt = multiprocessing.cpu_count()
    num_processes = math.floor(0.75 * cpu_cnt)

    # Setup Segment-Wise multiprocessing per PDF
    pdf_objs = []
    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path)

        logger.debug(f"Converting to Images with Async Parallel for PDF: {pdf_name} ...")
        pdf_obj = PDF(pdf_name)
        # Count the number of pages in PDF
        doc = pypdf.open(pdf_path)
        num_pages = doc.page_count
        # Total Processes should never be greater than number of processes
        procs = min(num_pages, num_processes)
        per_proc = math.ceil(num_pages / procs)

        # Generate the start-stop vectors
        vecs = []
        pdf_imgs_ord = []
        for i in range(num_processes):
            start = i * per_proc
            stop = min(num_pages, (i + 1) * per_proc)
            vecs.append((start, stop))
        # Parallely generate the PDF pages for each PDF
        with multiprocessing.Pool(processes=num_processes) as pool:
            pdf_imgs = pool.starmap(pdf_to_img, [(pdf_path, start, stop) for start, stop in vecs])
        pool.close()
        pool.join()
        for seg in pdf_imgs:
            pdf_imgs_ord.extend(seg)

        # Setup Page obj for the PDF
        for img in pdf_imgs_ord:
            orient = detect_text_orientation(img)
            if orient == "landscape":
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            page = Page(page_img=img, orient=orient)
            # Add the page to the PDF obj
            pdf_obj(page)

        pdf_objs.append(pdf_obj)

    return pdf_objs

    # print(f'Total Time taken: {time.time() - t0} seconds')
    # print(f'Number of PDF objs: {len(pdf_objs)}')
    # print(f'Number of Page objects in PDF 0: {len(pdf_objs[0].pages)}')
    # print(f'Image of Page 32 of PDF 1 : {pdf_objs[1].pages[31].page_img}')


def detect_regions(
    args: argparse.Namespace, pdfs: List[PDF], device: torch.device = torch.device("cuda:0")
) -> None:
    """Takes in PDF class objects, process the pages to generate molecule region detections"""
    logger.debug("Loading yolo weights...")
    model = YOLO(args.detect_weight)

    conf = 0.2
    iou_thresh = 0.4

    # Predict for each PDF
    for obj in pdfs:
        pdf_imgs = []
        for page in obj.pages:
            pdf_imgs.append(page.page_img)

        results = model.predict(pdf_imgs, conf=conf, iou=iou_thresh, device=device)

        # Store the boxes in the objects first
        # and then crop the required window for the molecule
        for i, img_results in enumerate(results):
            boxes = img_results.boxes
            page_box_list = []
            for box in boxes:
                if len(box.xyxy):
                    b = box.xyxy.cpu().numpy()[0].astype(np.int32)
                    page_box_list.append(b)
            page_boxes = np.array(page_box_list).reshape((-1, 4))
            # Compile the molecule images into the object
            # Sort the boxes by the y-axis:
            sort_idx = np.argsort(page_boxes[:, 1])
            page_boxes = page_boxes[sort_idx]
            # Add the page boxes and box images to the obj
            obj.pages[i].boxes = page_boxes

            page_box_imgs = []
            pg_img = pdf_imgs[i]
            if pg_img is None:
                continue
            for b in page_boxes:
                mol_img = pg_img[b[1] : b[3], b[0] : b[2], :]
                page_box_imgs.append(mol_img)
            obj.pages[i].box_images = page_box_imgs

    # DEBUG MODE Outputs: Overlaid Page Images with Boxes
    if args.debug:
        for obj in pdfs:
            for i, page in enumerate(obj.pages):
                if page.boxes is None or page.page_img is None:
                    continue
                page_img = copy.deepcopy(page.page_img)
                for j, box in enumerate(page.boxes):
                    if not args.tables:
                        page_img = add_box_and_text_over_image(page_img, box, str(j + 1))
                    else:
                        page_img = add_box_and_text_over_image(page_img, box)

                obj.pages[i].overlaid_img = page_img


def parse_molecules(args: argparse.Namespace, pdfs: List[PDF], device: torch.device) -> None:
    """Use the molecule images to generate the canonical SMILES string"""
    # Initialize the model with the checkpoint
    model = MolScribe(args.parser_weight, device=device)

    # Loop till batch is filled and then populate the SMILES
    for i, obj in enumerate(pdfs):
        logger.debug(f"Processing Molecules for PDF: {obj.name}")
        for j, page in tqdm(enumerate(obj.pages), total=len(obj.pages), desc="Processing Page:"):
            if len(page.boxes):
                page_molimgs = []
                page_smis = []
                page_confs = []
                for k, mol_img in enumerate(page.box_images):
                    act_in = cv2.cvtColor(mol_img, cv2.COLOR_BGR2RGB)
                    page_molimgs.append(act_in)

                # Get the predictions
                preds = model.predict_images(page_molimgs, return_confidence=True)

                for pred in preds:
                    pred_smi = pred["smiles"].split(".")
                    pred_smi = max(pred_smi, key=len)
                    pred_smi_can, success = canonicalize_smiles(pred_smi)
                    if success:
                        pred_smi = pred_smi_can
                    pred_conf = pred["confidence"]
                    page_smis.append(pred_smi)
                    page_confs.append(pred_conf)

                # Record the SMILES and confidences
                pdfs[i].pages[j].smiles = page_smis
                pdfs[i].pages[j].conf = page_confs


def generate_table_overlay(pdfs: List[PDF]) -> None:
    """Overlay the table metadata on the PDF images, if available"""
    for obj in pdfs:
        for i, page in enumerate(obj.pages):
            if page.boxes is None or page.overlaid_img is None or page.metadata is None:
                logger.debug(f"No boxes, overlaid_img or metadata found for Page: {i + 1}")
                continue
            page_img = copy.deepcopy(page.overlaid_img)
            for j, box in enumerate(page.boxes):
                if page.metadata[j] is not None:
                    metas = page.metadata[j]
                    str_meta = ""
                    for meta in metas:
                        if meta is not None:
                            for k in meta.keys():
                                str_meta += k + ":" + meta[k] + ","
                    if len(str_meta):
                        page_img = add_box_and_text_over_image(
                            page_img,
                            box,
                            str_meta,
                        )
                    else:
                        page_img = add_box_and_text_over_image(
                            page_img,
                            box,
                            f"BOX {j + 1} (no metadata)",
                        )
                else:
                    page_img = add_box_and_text_over_image(
                        page_img,
                        box,
                        f"BOX {j + 1} (no metadata)",
                    )
            # Overwrite the existing image
            obj.pages[i].overlaid_img = page_img
