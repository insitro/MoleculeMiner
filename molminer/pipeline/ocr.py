import copy
import logging
import re
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from doctr.models import ocr_predictor
from numpy import ndarray
from shapely.geometry import box
from tqdm import tqdm

from molminer.pipeline.data_structures import PDF

logger = logging.getLogger("molminer.pipeline.ocr")


def temp_viz(img: ndarray):
    cv2.imwrite("Temp_Viz.png", img)
    exit()


def detect_text_orientation(img: ndarray) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    dst = cv2.Canny(thresh, 50, 200, None, 3)
    # cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    tot_angles = 0
    for i in range(0, len(lines)):
        l = lines[i][0]  # noqa
        degree = np.abs(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])))
        tot_angles += degree
        # cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    avg_deg = tot_angles / len(lines)

    # Calculate the average angle to get the dominant text orientation
    if 0 <= avg_deg <= 45 or 135 < avg_deg < 180:
        return "portrait"
    elif 45 < avg_deg <= 135:
        return "landscape"
    else:
        raise Exception(f"Found an uncommon angle: {avg_deg}")


def link_refs(
    fil_words: List[str], fil_wordbbs: List[List[float]], mol_boxes: ndarray, orient: str
) -> List[Optional[str]]:
    """
    Constraints for linking molecule regions with available reference choices
    1. y2 of ref has to be within y1 and y2 of mol_box OR
        1.a If x1 of ref < x1 of mol_box
    2. x2 of ref has to be within x1 and x2 of mol_box
        2.a If If y2 of ref < y2 of mol_box
    5. Closest of the remaining choices of refs matched with mol_box by
        calculating distance between centroid mol_box and box. Matched IF
        (distance < cent_mol_box_x-x1) if (3.) satisfied OR
        (distance < cent_mol_box_y-y1) if (4.) satisfied
    """
    refs: List[Optional[str]] = []
    word_polys = [box(*b) for b in fil_wordbbs]
    # Sort mol boxes by y
    # mol_boxes = np.array(mol_boxes).reshape((-1,4))
    # sort_idx = np.argsort(mol_boxes[:,1])
    # mol_boxes = mol_boxes[sort_idx]
    pattern = None
    for idx_mol, mol_box in enumerate(mol_boxes):
        box_poly = box(*mol_box)
        cur_fil = []
        cur_sides = []
        cur_dists = []

        for idx, word_bb in enumerate(fil_wordbbs):
            # print('WORD')
            # print(mol_box[1], word_bb[3], mol_box[3])
            # print(mol_box[0], word_bb[2], mol_box[2])
            # Change vertex comparison based on orientation
            if orient == "portrait":
                comp_y = word_bb[3]
                comp_x = word_bb[2]
            else:
                comp_y = word_bb[1]
                comp_x = word_bb[0]
            if mol_box[1] - 100 < comp_y < mol_box[3] + 80:
                # print('COND 1 Satisfied')
                if word_bb[0] <= mol_box[0]:
                    dist = word_polys[idx].distance(box_poly)
                    # if dist < abs(mol_box[2] - mol_box[0]) or \
                    #     dist < abs(mol_box[3] - mol_box[1]):
                    if dist < 400:
                        # print('COND 1.1 Satisfied')
                        cur_fil.append(idx)
                        cur_sides.append("Left")
                        cur_dists.append(dist)
            if mol_box[0] - 100 < comp_x < mol_box[2]:
                # print('COND 2 Satisfied')
                if word_bb[3] >= mol_box[3]:
                    dist = word_polys[idx].distance(box_poly)
                    # if dist < abs(mol_box[2] - mol_box[0]) or \
                    #     dist < abs(mol_box[3] - mol_box[1]):
                    if dist < 200:
                        # print('COND 1.1 Satisfied')
                        cur_fil.append(idx)
                        cur_sides.append("Bot")
                        cur_dists.append(dist)
        if len(cur_fil):
            if len(cur_dists):
                min_dist = min(cur_dists)
                min_idx_pos = cur_dists.index(min_dist)
                min_idx = cur_fil[min_idx_pos]
                # import pdb; pdb.set_trace()
                # Find the pattern on the first iteration
                if pattern is None:
                    pattern = cur_sides[min_idx_pos]
                refs.append(fil_words[min_idx])
            else:
                refs.append(None)
        else:
            refs.append(None)
    return refs


def rotate_boxes(word_bbs: List[List[float]], h: float, w: float):
    """
    Function to rotate the detected Word Boxes from the OCR
    """
    rot_boxes = []
    for wbox in word_bbs:
        x1, y1 = wbox[1], w - wbox[2]
        x2, y2 = wbox[3], w - wbox[0]
        rot_boxes.append([x1, y1, x2, y2])
    return rot_boxes


def find_page_refs(
    words: List[str],
    words_bb: List[List[float]],
    mol_boxes: ndarray,
    orient: str,
    height: float,
    width: float,
    # page_img: Optional[ndarray] = None,
):
    # Change the box coords if orientation is landscape
    if orient == "landscape":
        words_bb = rotate_boxes(words_bb, height, width)
        # page_img = cv2.rotate(page_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Create polygons of all molBBs and wordBBs
    mol_polys = [box(vals[0], vals[1], vals[2], vals[3]) for vals in mol_boxes]
    word_polys = [box(vals[0], vals[1], vals[2], vals[3]) for vals in words_bb]

    # Remove all the invalid WORDs
    # (ie those that fall within the bounds of the molecule box itself)
    valid_wordbbs = []
    valid_words = []
    added_idxs = []
    for mol_poly in mol_polys:
        for idx, word_poly in enumerate(word_polys):
            inter = word_poly.intersection(mol_poly).area
            union = word_poly.union(mol_poly).area
            iou = inter / union

            if iou != 0:
                if idx not in added_idxs:
                    added_idxs.append(idx)

    # Collect all the words that did not intersect
    for idx, word in enumerate(words):
        if idx not in added_idxs:
            valid_words.append(word)
            valid_wordbbs.append(words_bb[idx])

    # Remove any Pattern that does not match a reference number
    fil_words = []
    fil_wordbbs = []
    pattern = r"^[A-Za-z]?\-?[0-9]{1,4}\.?$"
    for idx, word in enumerate(valid_words):
        filtrd = re.findall(pattern, word)
        if len(filtrd):
            fil_words.append(word)
            fil_wordbbs.append(valid_wordbbs[idx])

    # Link each molecule region with its corresponding reference no
    mol_refs = link_refs(fil_words, fil_wordbbs, mol_boxes, orient)
    # visualize_twin_overlays(page_img, fil_words, fil_wordbbs, mol_boxes,
    # refs=mol_refs)
    return mol_refs


def find_references(pdfs: List[PDF]):
    # Initialize the OCR
    device = torch.device("cuda:0")
    model = ocr_predictor("db_resnet50", pretrained=True, straighten_pages=True).to(device)

    for pdf in pdfs:
        logger.debug(f"Processing References for PDF: {pdf.name}")
        pdf_pages: List[Union[ndarray, None]] = []
        pdf_pages_molboxes = []
        pdf_pages_orient = []
        for page in pdf.pages:
            if page.page_img is None:
                pdf_pages.append(page.page_img)
                pdf_pages_molboxes.append(page.boxes)
                pdf_pages_orient.append("portrait")
                continue

            # Detect Orientation of Page
            orient = detect_text_orientation(page.page_img)
            if orient == "portrait":
                pdf_pages.append(page.page_img)
            else:
                page_img = cv2.rotate(page.page_img, cv2.ROTATE_90_CLOCKWISE)
                pdf_pages.append(page_img)
            pdf_pages_molboxes.append(page.boxes)
            pdf_pages_orient.append(orient)

        pages_refs = []

        for i, pdf_page in tqdm(
            enumerate(pdf_pages), desc="Processing Page: ", total=len(pdf_pages)
        ):
            # Only process pages that have molecules
            if isinstance(pdf_pages_molboxes[i], np.ndarray):
                # Pass the Images through the OCR
                result = model([pdf_page])
                result = result.export()
                # words_dic = result['pages'][0]['blocks'][0]['lines'][0]['words']

                height, width = (
                    result["pages"][0]["dimensions"][0],
                    result["pages"][0]["dimensions"][1],
                )
                words = []
                words_bb = []
                for block in result["pages"][0]["blocks"]:
                    for line in block["lines"]:
                        for word in line["words"]:
                            if len(word["geometry"]) != 5:
                                (xmin, ymin), (xmax, ymax) = word["geometry"]  # (_,_), (_,_),
                                xmin, ymin = int(xmin * width), int(ymin * height)
                                xmax, ymax = int(xmax * width), int(ymax * height)
                                words.append(word["value"])
                                words_bb.append([xmin, ymin, xmax, ymax])

                # Process the boxes and get the reference numbers
                page_mol_refs = find_page_refs(
                    words,
                    words_bb,
                    pdf_pages_molboxes[i],
                    pdf_pages_orient[i],
                    height,
                    width,
                    # pdf_page,
                )
                # # Visualize all identified words
                # if logger.level <= logging.DEBUG:
                #     bn = os.path.splitext(pdf.name)[0]
                #     fldr = f"outputs/{bn}/ref_overlays"
                #     os.makedirs(fldr, exist_ok=True)
                #     visualize_twin_overlays(
                #         pdf_page,
                #         words,
                #         words_bb,
                #         pdf_pages_molboxes[i],
                #         file_path=os.path.join(fldr, f"RefOverlay_Page_{i:05d}.png"),
                #         refs=page_mol_refs,
                #     )
                if any(ref is not None for ref in page_mol_refs):
                    pages_refs.append(page_mol_refs)
                else:
                    pages_refs.append(None)
            else:
                pages_refs.append(None)

        # Add the page refs to the PDF object
        for i, page in enumerate(pdf.pages):
            page.ref_nos = pages_refs[i]
    del model


def add_box_and_text_over_image(
    page_img: np.ndarray,
    box: np.ndarray,
    message: Optional[str] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    font_scale: float = 1.0,
    thickness: int = 1,
) -> np.ndarray:
    start = (box[0], box[1])
    end = (box[2], box[3])
    start_txt = (box[0], box[1] - 20)
    page_img = cv2.rectangle(page_img, start, end, (0, 0, 255), 2)
    if message is not None:
        page_img = cv2.putText(
            page_img,
            message,
            start_txt,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return page_img


def visualize_twin_overlays(
    page_img: ndarray,
    words: List[str],
    words_bb: List[List[int]],
    mol_bb: ndarray,
    refs: Optional[List[str]] = None,
    file_path: Optional[str] = None,
) -> None:
    page_img = copy.deepcopy(page_img)
    page_boxes = mol_bb
    for j, pbox in enumerate(page_boxes):
        if refs is None:
            page_img = add_box_and_text_over_image(page_img, pbox, str(j + 1))
        else:
            page_img = add_box_and_text_over_image(page_img, pbox, f"{j + 1} - {refs[j]}")

    for j, wbox in enumerate(words_bb):
        page_img = add_box_and_text_over_image(page_img, np.array(wbox), words[j])

    if file_path is not None:
        cv2.imwrite(file_path, page_img)
    else:
        cv2.imwrite("Test_TwinOverlay.png", page_img)
