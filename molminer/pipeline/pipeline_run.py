"""
********************* MOLECULE DATA EXTRACTION PIPELINE *********************
AUTHOR: Abhisek Dey (Summer Intern, 2024)

Main Pipeline that converts PDF(s) into a CSV file containing PDF level, Page Level:
           ********** FIXED OUTPUTS **********
A CSV file per PDF containing the following:

1. Bounding Box for Each Detected Molecule Region on a 300 DPI converted image of the page
2. Associated Canonical SMILES string for the detected molecule
3. Associated Confidence Score for each molecule parsed by the parser
           ********** OPTIONAL OUTPUTS (enabled with --debug flag)**********
1. PDF pages converted to 300 DPI
2. Overlaid detected molecule regions on the PDF pages
3. Individual molecule region images


"""

import argparse
import logging
import os
import sys
import time
from typing import List, Tuple

import cv2
import torch
from tqdm import tqdm

from molminer.gpt_tables.detections import GPT_Table
from molminer.pipeline.data_structures import (
    PDF,
    TSV_HEADER,
)
from molminer.pipeline.link_final import link_tables
from molminer.pipeline.ocr import find_references
from molminer.pipeline.utils import (
    convert_pdfs,
    detect_regions,
    generate_table_overlay,
    parse_molecules,
)


def write_tsv(args: argparse.Namespace, pdf: PDF) -> Tuple[int, int]:
    # Write the CSV for the object
    name = os.path.splitext(pdf.name)[0]
    pdf_csv_dir = os.path.join(args.out_dir, name)
    os.makedirs(pdf_csv_dir, exist_ok=True)

    df = pdf.build_df()
    df.to_csv(os.path.join(pdf_csv_dir, "mol_smiles.csv"), sep="\t", index=False)

    tot_mols = len(df)
    meta_cols = [col for col in df.columns if col not in TSV_HEADER]
    tot_meta = df[meta_cols].dropna(axis=0, how="all").shape[0]
    return tot_mols, tot_meta


# Write the final CSV
def write_all_tsv(args: argparse.Namespace, pdfs: list, logger: logging.Logger) -> None:
    tot_mols = 0
    tot_meta = 0
    # Write the CSV for the object
    for obj in pdfs:
        tmp_tot_mols, tmp_tot_meta = write_tsv(args, obj)
        tot_mols += tmp_tot_mols
        tot_meta += tmp_tot_meta
    logger.info(
        f"\n WRITTEN ALL PDF CSVs   |    TOTAL MOLECULES - {tot_mols}     |"
        f"      TOTAL MOLECULES WITH METADATA - {tot_meta}\n"
    )


# Write the overlaid page images per PDF (DEBUG Mode Only)
def write_extra(args: argparse.Namespace, pdfs: List[PDF]) -> None:
    for obj in pdfs:
        name = os.path.splitext(obj.name)[0]
        pdf_img_dir = os.path.join(args.out_dir, name, "overlaid_pages")
        if not os.path.exists(pdf_img_dir):
            os.makedirs(pdf_img_dir)

        # If Table Mode, generate table metadata as Annotation Overlay
        if args.tables:
            generate_table_overlay(pdfs)

        for i, page in tqdm(
            enumerate(obj.pages), total=len(obj.pages), desc="Processing Pages (DEBUG)"
        ):
            if len(page.boxes):
                cv2.imwrite(
                    os.path.join(pdf_img_dir, f"Page_{i + 1:0>{5}}.png"), page.overlaid_img
                )

    print("\n WRITTEN DEBUG OVERLAID IMAGES \n")


def main_process(args: argparse.Namespace, logger: logging.Logger) -> None:
    start = time.time()
    # *************** PHASE - 1 (Convert PDF to IMAGES) ******************
    logger.info("\n PHASE - 1 (PDF to IMAGES) \n")
    pdfs = convert_pdfs(args)
    end1 = time.time()
    # *************** PHASE - 2 (Convert PDF to IMAGES) ******************
    if args.api_key is None or args.api_key == "":
        args.api_key = os.environ.get("OPENAI_API_KEY")
    if args.tables and args.api_key:
        logger.info("\n PHASE - 2 (IMAGES to TABLES) \n")
        table_ext = GPT_Table(args.api_key)
        table_ext.process_pdfs(pdfs)
    else:
        logger.info("Skipping Table Extraction Phase - Not requested or no API Key provided")
    end2 = time.time()
    # *************** PHASE - 3 (Pages to Molecule Regions) **************
    logger.info(" \n PHASE - 3 (DETECTION) \n")
    if torch.cuda.is_available():
        logger.info("Found at least one CUDA Device, using GPU for Detection Phase")
        device = torch.device("cuda")
    else:
        logger.info("Did not find any GPU device, defaulting to CPU for Detection Phase")
        device = torch.device("cpu")

    detect_regions(args, pdfs, device=device)
    end3 = time.time()

    # *************** PHASE - 3a (Find the reference Number) **************
    logger.info(" \n PHASE - 3a (MOLECULE REFERENCES) \n")
    if args.tables:
        find_references(pdfs)
    end3a = time.time()

    # *************** PHASE - 4 (Parse Molecule Images into SMILES) **************
    logger.info(" \n PHASE - 4 (PARSING) \n")
    parse_molecules(args, pdfs, device=device)
    end4 = time.time()

    # *************** PHASE - 5 (Parse Molecule Images into SMILES) **************
    if args.tables:
        logger.info(" \n LINKING molecules with table data if any")
        link_tables(pdfs)
    end5 = time.time()

    # *************** PHASE - 6 (Write the output as a CSV) **************
    logger.info("PHASE 5 - WRITING Final CSV File")
    logger.debug(f"Length of PDFs: {len(pdfs)}")
    write_all_tsv(args, pdfs, logger)
    end6 = time.time()

    if args.debug:
        logger.info("DEBUG MODE SET; Writing the Overlaid Pages")
        write_extra(args, pdfs)
    end7 = time.time()

    # Printing Time Stats
    logger.info("\n TIME STATS \n")
    if args.debug:
        logger.info(f"{'1. PDF to Images':40} : {end1 - start}secs")
        logger.info(f"{'2. Table Detection & Extraction':40} : {end2 - end1}secs")
        logger.info(f"{'3. Molecule Figures Detection':40} : {end3 - end2}secs")
        logger.info(f"{'3a. Molecule Reference Extraction':40} : {end3a - end3}secs")
        logger.info(f"{'4. Molecule Images to SMILES':40} : {end4 - end3a}secs")
        logger.info(f"{'5. Molecule to Table Data Linking':40} : {end5 - end4}secs")
        logger.info(f"{'6. Writing Final CSV':40} : {end6 - end5}secs")
        logger.info(f"{'7. DEBUG: Writing Overlaid Images':40} : {end7 - end6}secs")
        logger.info(f"{'TOTAL Elapsed Time':40} : {end7 - start}secs")


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Molecule Data Extraction Pipeline")
    # DATA in; DATA out
    parser.add_argument(
        "--in_pdfs",
        default="inputs/test_pdfs",
        type=str,
        help="Path to the Directory containing list of PDFs OR a single PDF",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs",
        type=str,
        help="Output Directory for saving the generated CSV files and optionally debug files",
    )
    # LOGGING MODE
    logmode_options = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument(
        "--logmode",
        choices=logmode_options,
        default="INFO",
        help=(
            "Mode for logging while running the app. Options are "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL"
        ),
        type=str,
    )
    # EXTRACT TABLES
    parser.add_argument(
        "--tables",
        action="store_true",
        help="If specified, will look for tables in the PDFs and will link them to the molecules",
    )
    # WEIGHT paths
    parser.add_argument(
        "--detect_weight",
        default="weights/detection.pt",
        type=str,
        help="Absolute path to the YOLO detection model weights",
    )
    parser.add_argument(
        "--parser_weight",
        default="weights/parsing.pth",
        type=str,
        help="Absolute path to the MolscribeV2 model weights",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        type=str,
        help="Your OpenAI API key for GPT-4 Table Extraction",
    )
    return parser


def run_pipeline():
    parser = parse_args()
    args = parser.parse_args()
    if args.logmode.upper() == "DEBUG":
        args.debug = True

    logger = logging.getLogger("molminer")
    logger.setLevel(logging.getLevelName(args.logmode.upper()))
    # Disable propagation that would bubble up to other loggers, if any
    logger.propagate = False
    # create file handler which logs even debug messages
    fh = logging.FileHandler("molminer.log")
    # create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Start the stages
    main_process(args, logger)


if __name__ == "__main__":
    run_pipeline()
