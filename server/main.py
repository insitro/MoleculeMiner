import base64
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from redun import File as RedunFile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from molminer.gpt_tables.detections import GPT_Table
from molminer.pipeline.link_final import link_tables
from molminer.pipeline.ocr import find_references
from molminer.pipeline.pipeline_run import parse_args, write_extra
from molminer.pipeline.utils import (
    convert_pdfs,
    detect_regions,
    parse_molecules,
)

PROCESSED_FILE_DIR = "processed_files"


def list_processed_files(glob_pattern: str = "*.pdf") -> List[str]:
    files = glob.glob(f"{PROCESSED_FILE_DIR}/{glob_pattern}")
    files = [os.path.basename(f) for f in files]
    return files


def fetch_processed_filenames(filename: str) -> Tuple[Optional[str], Optional[str]]:
    avail_pdfs = list_processed_files()
    avail_base_fn = [os.path.splitext(os.path.basename(f))[0] for f in avail_pdfs]
    expected_base_fn = os.path.splitext(os.path.basename(filename))[0]
    # This should never happen but just in case
    if expected_base_fn not in avail_base_fn:
        logger.debug(f"Expected basename {expected_base_fn} not in avail_pdfs: {avail_pdfs}")
        return None, None
    csv_file = os.path.join(PROCESSED_FILE_DIR, f"{expected_base_fn}.csv")
    pdf_file = os.path.join(PROCESSED_FILE_DIR, f"{expected_base_fn}.pdf")
    if not os.path.exists(csv_file) or not os.path.exists(pdf_file):
        logger.debug(f"One of the files {csv_file} or {pdf_file} does not exist; returning None")
        return None, None
    return csv_file, pdf_file


os.makedirs(PROCESSED_FILE_DIR, exist_ok=True)

parser = parse_args()
args = parser.parse_args()
if args.logmode.upper() == "DEBUG":
    args.debug = True
else:
    args.debug = False

logger = logging.getLogger("molminer")
logger.setLevel(logging.getLevelName(args.logmode.upper()))
# Clear any existing handlers to avoid duplication
logger.handlers.clear()
# Disable propagation that would bubble up to other loggers, if any
logger.propagate = False
# create file handler
fh = logging.FileHandler("molminer_app.log")
fh.setLevel(logging.getLevelName(args.logmode.upper()))
# create console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.getLevelName(args.logmode.upper()))
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

local_detect_wts = RedunFile("weights/detection.pt")
s3_det_wts = RedunFile("s3://2025-molecule-miner/weights/detection.pt")
if not local_detect_wts.exists():
    s3_det_wts.copy_to(local_detect_wts)

local_parser_wts = RedunFile("weights/parsing.pth")
s3_parser_wts = RedunFile("s3://2025-molecule-miner/weights/parsing.pth")
if not local_parser_wts.exists():
    s3_parser_wts.copy_to(local_parser_wts)


app = FastAPI()

# Create directories for static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/server/static", StaticFiles(directory="server/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="server/templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "home"})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request, "active_page": "upload"})


@app.get("/viewer", response_class=HTMLResponse)
async def list_files(request: Request):
    files = list_processed_files()
    return templates.TemplateResponse(
        "viewer.html", {"request": request, "files": files, "active_page": "viewer"}
    )


@app.get("/view_file")
async def view_file(filename: str):
    logger.debug(f"Viewing file: {filename}")
    files = list_processed_files()
    csv_file, pdf_file = fetch_processed_filenames(filename)
    if csv_file is None or pdf_file is None:
        logger.debug(
            f"Could not find any processed files for {filename}; Available files: {files}"
        )
        return {
            "message": "Could not find any processed files for the given PDF",
            "pdf_name": filename,
            "csv_file": None,
        }

    df = pd.read_csv(csv_file, sep="\t")

    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()

    return {
        "message": "Files processed successfully",
        "pdf_file": base64.b64encode(pdf_bytes).decode(),
        "csv_file": df.to_csv(index=False, sep="\t").encode("utf-8"),
    }


async def process_file_in_background(
    csv_file_path, args, table_extraction: bool, api_key: Optional[str] = None
):
    pdfs = convert_pdfs(args)

    if api_key is None or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is not None and table_extraction:
        logger.info("PHASE 2 - IMAGES to TABLES")
        args.tables = True
        table_ext = GPT_Table(api_key)
        table_ext.process_pdfs(pdfs)
    else:
        # Args is global so we must set it to False here to not
        # affect future runs
        args.tables = False
        if api_key is None:
            logger.debug(f"API key info: {api_key}; type: {type(api_key)}")
            logger.info("No API key provided for table extraction; skipping")
        else:
            logger.debug(f"Extraction value: {table_extraction}")
            logger.info("Skipping table extraction b/c user did not request it")

    logger.info("PHASE 3 - DETECTION")
    if torch.cuda.is_available():
        logger.info("Found at least one CUDA Device, using GPU for Detection Phase")
        device = torch.device("cuda")
    else:
        logger.info("Did not find any GPU device, defaulting to CPU for Detection Phase")
        device = torch.device("cpu")

    detect_regions(args, pdfs, device=device)

    logger.info("PHASE 3a - MOLECULE REFERENCES")
    if args.tables:
        find_references(pdfs)

    logger.info("PHASE 4 - PARSING")
    parse_molecules(args, pdfs, device=device)
    if args.tables:
        logger.info("LINKING molecules with table data if any")
        link_tables(pdfs)

    logger.info("PHASE 5 - WRITING Final CSV File")
    df = pdfs[0].build_df()
    df.to_csv(csv_file_path, index=False, sep="\t")

    if args.debug:
        logger.debug("DEBUG MODE SET; Writing the Overlaid Pages")
        write_extra(args, pdfs)


@app.post("/process")
async def process_files(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    api_key: Optional[str] = Form(None),
    table_extraction: bool = Form(False),
):
    logger.info("PHASE 1 - LOADING PDF")
    logger.debug(f"Extracting PDF file: {pdf_file.filename}")
    logger.debug(f"Using API key: {api_key}")
    logger.debug(f"User submitted table_extraction value: {table_extraction}")

    pdf_file_name = pdf_file.filename
    if pdf_file_name is None or pdf_file_name == "":
        return {"message": "No PDF file uploaded"}
    if not pdf_file_name.endswith(".pdf"):
        pdf_file_name = pdf_file_name + ".pdf"
    csv_file_name = pdf_file_name.replace(".pdf", ".csv")
    pdf_file_path = os.path.join(f"{PROCESSED_FILE_DIR}", pdf_file_name)
    csv_file_path = os.path.join(f"{PROCESSED_FILE_DIR}", csv_file_name)

    # If the file has already been processed, return the CSV file
    if os.path.exists(csv_file_path):
        return {
            "message": "Files processed successfully",
            "pdf_name": pdf_file.filename,
            "csv_file": csv_file_path,
        }

    bytes_data = await pdf_file.read()
    with open(pdf_file_path, "wb") as f:
        f.write(bytes_data)
    args.in_pdfs = pdf_file_path

    # Add the processing to background tasks
    background_tasks.add_task(
        process_file_in_background, csv_file_path, args, table_extraction, api_key
    )

    return {
        "message": "File uploaded successfully. Processing started.",
        "pdf_name": pdf_file.filename,
        "status": "processing",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7887, log_level="debug" if args.debug else "info")
