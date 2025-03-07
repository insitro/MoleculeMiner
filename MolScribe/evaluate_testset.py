# Changed for core
import argparse
import csv
import math
import os
import warnings
from typing import Tuple

import torch
from evaluate import SmilesEvaluator
from Levenshtein import distance
from tqdm import tqdm

from molscribe import MolScribe

warnings.filterwarnings("ignore")


def compile_pairs(args):
    # Initialize the Model
    model = MolScribe(args.ckpt_path, device=torch.device("cuda:0"))
    pdfs = os.listdir(args.csv_dir)
    pdfs = [pdf.split(".")[0] for pdf in pdfs]

    gts = []
    preds = []

    for pdf in pdfs:
        pdf_csv_path = os.path.join(args.csv_dir, pdf + ".csv")
        # Read in the GT SMILES
        with open(pdf_csv_path, "r") as file:
            csvfile = csv.DictReader(file)
            for line in csvfile:
                gts.append(line["SMILES"])

        # Populate the paths for all the images
        pdf_img_path = os.path.join(args.pdf_mol_imgs, pdf + ".pdf")
        pdf_imgs = os.listdir(pdf_img_path)
        pdf_imgs_paths = sorted([os.path.join(pdf_img_path, pdf_img) for pdf_img in pdf_imgs])

        # Predict the output SMILES:
        print(f"Predicting SMILES for the molecules of {pdf}")
        for i in tqdm(
            range(0, len(pdf_imgs_paths), 32),
            total=math.ceil(len(pdf_imgs_paths) / 32),
            desc="Processing batch:",
        ):
            batch_img_paths = pdf_imgs_paths[i : min(i + 32, len(pdf_imgs_paths))]
            batch_out = model.predict_image_files(batch_img_paths)
            for out in batch_out:
                pred_smi = out["smiles"].split(".")
                pred_smi = max(pred_smi, key=len)
                preds.append(pred_smi)

    lev, errors, error_ids = compute_levenshtein(preds, gts)
    print(f"Mean Levenstein Distance: {lev}")

    # Now send the smiles for evaluation
    print("Running Complete Eval...")
    eval_fn = SmilesEvaluator(gold_smiles=gts, num_workers=8)
    results = eval_fn.evaluate(pred_smiles=preds)
    print(results)

    # Write the results
    with open("PredVSGT.txt", "w") as f:
        for i in range(len(preds)):
            f.write(preds[i] + "," + gts[i] + "\n")
    f.close()
    print("Written Total Prediction File")

    with open("Errors.txt", "w") as f:
        for i in range(len(error_ids)):
            f.write(str(error_ids[i]) + "\t" + errors[i][0] + "\t" + errors[i][1] + "\n")
    f.close()
    print("Written Error File")


def compute_levenshtein(preds, gts) -> Tuple[float, list, list]:
    tot_lev = 0
    errors = []
    error_ids = []
    for i in range(len(preds)):
        dist = distance(preds[i], gts[i])
        tot_lev += dist
        if dist > 0:
            errors.append([preds[i], gts[i]])
            error_ids.append(i + 1)
    mean_lev = tot_lev / len(preds)
    return mean_lev, errors, error_ids


def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Insitro Molecul Dataset Evaluation \
                                     for Own Patent Data"
    )
    parser.add_argument(
        "--csv_dir",
        default="/home/ec2-user/my_work/chem_data_extraction/detection/MoleculeBank/csvs",
        type=str,
        help="CSV Directory containing the CID-SMILES annotation/PDF",
    )
    parser.add_argument(
        "--pdf_mol_imgs",
        default="/home/ec2-user/my_work/chem_data_extraction/detection/MoleculeBank",
        type=str,
        help="Directory containing the molecule Images per PDF",
    )
    parser.add_argument(
        "--ckpt_path",
        default="swin_base_char_aux_1m.pth",
        type=str,
        help="Path to the saved checkpoint model",
    )

    args = parser.parse_args()

    # Compile the Pred-GT Pairs from all available PDFs
    compile_pairs(args)


if __name__ == "__main__":
    parse_args()
