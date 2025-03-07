"""
Custom Utils to Read and Compute Tanimoto Similarity and Graph-Edit Distances

Author - Abhisek Dey
"""

from collections import defaultdict
from typing import List

import networkx as nx
import timeout_decorator
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


# Expects an RDKit Molecule as argument
def construct_mol_graph(rd_mol) -> nx.DiGraph:
    # Mol Graph
    g = nx.DiGraph()
    atom_cnts = defaultdict(lambda: 1)  # type: ignore
    atom_idx2lbl = {}
    # Create all the nodes first
    for atom in rd_mol.GetAtoms():
        atom_lbl = atom.GetSymbol()
        atom_idx = atom.GetIdx()
        idx = atom_cnts[atom_lbl]
        atom_cnts[atom_lbl] += 1
        node_name = atom_lbl + "_" + str(idx)
        atom_idx2lbl[atom_idx] = node_name
        g.add_node(node_name)

    # Create all the edges now
    for bond in rd_mol.GetBonds():
        strt_atom = atom_idx2lbl[bond.GetBeginAtomIdx()]
        end_atom = atom_idx2lbl[bond.GetEndAtomIdx()]
        bond_type = bond.GetBondType()
        g.add_edge(strt_atom, end_atom, btype=bond_type)

    return g


def node_match(n1, n2):
    return n1 == n2


def edge_match(e1, e2):
    return e1["btype"] == e2["btype"]


# Reads in Pred and GT SMILES, Creates an RDKIT Molecule and computes the graph edit distance
# NOTE: Due to GED being a NP hard problem some comparisons can get stuck. This is a timeout based
# workaround. Time increases exponentially with the number of nodes
@timeout_decorator.timeout(5)
def do_ged(pred_graph, gt_graph):
    try:
        gedist = nx.graph_edit_distance(pred_graph, gt_graph, edge_match=edge_match)
        return gedist
    except timeout_decorator.timeout_decorator.TimeoutError:
        return None


def compute_graph_edit(preds, gts):
    dists = []
    for i in tqdm(range(len(preds)), total=len(preds), desc="Processing Molecules..."):
        try:
            pred_mol = Chem.MolFromSmiles(preds[i])
            gt_mol = Chem.MolFromSmiles(gts[i])
            pred_graph = construct_mol_graph(pred_mol)
            gt_graph = construct_mol_graph(gt_mol)

            gedist = do_ged(pred_graph, gt_graph)
            if gedist is None:
                continue
            else:
                dists.append(gedist)
        except:  # noqa: E722
            continue

    print(f"Mean Graph Edit Distsance: {sum(dists)/len(dists)}")
    print(f"Succesfully computed molecules: {len(dists)}")


def compute_tanimoto(pred_list, gt_list):
    fpgen = AllChem.GetMorganGenerator(radius=4)
    all_sims = []
    # First Filter Out Invalid SMILES and compute Similarity for Valid Ones
    for i in range(len(pred_list)):
        try:
            pred_mol = Chem.MolFromSmiles(pred_list[i])
            gt_mol = Chem.MolFromSmiles(gt_list[i])
            fp_pred = fpgen.GetFingerprint(pred_mol)
            fp_gt = fpgen.GetFingerprint(gt_mol)
            sim = DataStructs.TanimotoSimilarity(fp_pred, fp_gt)
            all_sims.append(sim)
        except:  # noqa: E722
            print(f"Invalid SMILES: {pred_list[i]}   OR    {gt_list[i]}")

    print(f"Mean Tanimoto Similarity on Morgan Fingerprints: {sum(all_sims)/len(all_sims)}")
    print(f"Total Molecules Compared: {len(all_sims)}")


# Expects input of a path to text file containing PredVSGT
def read_predicted(file_path) -> tuple[List, List]:
    preds = []
    gts = []

    data_file = open(file_path, "r")
    lines = data_file.readlines()

    for line in lines:
        row = line.rstrip("\n").split(",")
        preds.append(row[0])
        gts.append(row[1])

    return preds, gts


# Custom Script Test
if __name__ == "__main__":
    path = "PredVSGT.txt"
    pred_list, gt_list = read_predicted(path)
    compute_graph_edit(pred_list, gt_list)
    compute_tanimoto(pred_list, gt_list)
