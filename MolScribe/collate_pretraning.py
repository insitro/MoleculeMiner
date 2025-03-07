# Changed for core
"""
Short Script to Collate Pre-Training SMILES compilation

ARGS:
1. --data_dir: Directory containing the folders containing their csvs
2. --out_dir: Directory to store the collated data

EXAMPLE USAGE:
python collate_pretraining.py --data_dir data/ --out_dir compiled
"""

import argparse
import csv
import os

from rdkit import Chem
from tqdm import tqdm

OUTS = set(["compiled", "full_compiled", "final_csvs", "all_smiles"])


def collate(args):
    all_smiles = set()
    for i, ddir in enumerate(args.data_csvs):
        with open(ddir, "r") as file:
            csvFile = csv.DictReader(file)
            for line in tqdm(csvFile):
                smi = line[args.csv_keys[i]]
                can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                all_smiles.add(can_smi)
        file.close()

    print(f"Total Unique SMILES: {len(all_smiles)}")
    # Change it to a list for writing
    all_smiles = list(all_smiles)

    # Write the compiled list
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_path = os.path.join(args.out_dir, "all_canonicalized.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(all_smiles))
    print("Written all SMILES")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_csvs",
        type=str,
        default=["data/pubchem/train_1m.csv", "data/mos_zinc_chembl/allmolgen.csv"],
        nargs="+",
        help="Path to single CSVs",
    )
    parser.add_argument(
        "--csv_keys",
        type=str,
        default=["SMILES", "smiles"],
        nargs="+",
        help="Column names for the SMILES string",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/full_compiled",
        help="Path containing directories to the CSVs",
    )
    args = parser.parse_args()
    collate(args)


if __name__ == "__main__":
    parse_args()
