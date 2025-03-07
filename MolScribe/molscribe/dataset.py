# Changed for Core
import copy
import os
import random
import re
import string
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from molscribe.augment import CropWhite, PadWhite, SafeRotate, SaltAndPepperNoise
from molscribe.chemistry import normalize_nodes
from molscribe.constants import COLORS, ELEMENTS, RGROUP_SYMBOLS, SUBSTITUTIONS
from molscribe.indigo import Indigo
from molscribe.indigo.renderer import IndigoRenderer
from molscribe.tokenizer import PAD_ID
from molscribe.utils import FORMAT_INFO
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_FUNCTIONAL_GROUP_PROB = 0.8
INDIGO_CONDENSED_PROB = 0.5
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.8
INDIGO_COLOR_PROB = 0.2


# Custom Function to generate pixel maps
class GenerateMaps(object):
    def __init__(self, input_size, downscaling_factors):
        self.input_size = input_size
        self.downscaling_factors = downscaling_factors
        # Get the actual sizes for reshaping
        sizes = []
        cur_size = input_size
        for f in self.downscaling_factors:
            cur_size = cur_size // f
            sizes.append(cur_size)
        self.sizes = sizes
        self.map_transform = transforms.ToTensor()

    def __call__(self, img):
        # First Convert the Image into its pixel map
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img <= 250] = 0
        # img_label, _, _ = line_detector(img, "full",
        #                                        discontinuity_relative=15, bucket_size=64,
        #                                        minimum_for_fusion=1, llumi=110,
        #                                        blumi=110)
        thresh_pix = np.percentile(img, 1, method="interpolated_inverted_cdf")
        img[img <= thresh_pix] = 0
        img[img > thresh_pix] = 255

        img_label = cv2.bitwise_not(img)

        # img_label[img_label != 0] = 255
        # img_label = img_label.astype(np.uint8)
        # Now convert to the requisite sizes
        maps = []
        for f in self.sizes:
            img_label = cv2.resize(img_label, (f, f))
            img_label_copy = copy.deepcopy(img_label)
            img_label_copy[img_label_copy != 0] = 255
            img_label_copy = img_label_copy.reshape((f, f, 1)).astype(np.uint8)

            img_label_copy = self.map_transform(img_label_copy)
            maps.append(img_label_copy.squeeze())
        return maps


# Additional augmentations to account for rogue margins and cut off edges
class GenerateRealPerturbations(object):
    def __init__(self, perturb=False, perturb_ratio=0.02):
        self.perturb = perturb
        self.perturb_ratio = perturb_ratio

    def __call__(self, img):
        if self.perturb:
            h, w, _ = img.shape
            # Choose whether or not to perturb first
            if random.random() < 0.9:
                side = random.choice(range(4))
                # Choose whether crop or add bar
                if random.random() > 0.5:
                    # Crop 5% of height/width
                    if side == 0:
                        img = img[: int((1 - self.perturb_ratio) * h), :, :]
                    elif side == 1:
                        img = img[int(self.perturb_ratio * h) :, :, :]
                    elif side == 2:
                        img = img[:, : int((1 - self.perturb_ratio) * w), :]
                    else:
                        img = img[:, int(self.perturb_ratio * w) :, :]
                else:
                    # Add Bar
                    if side == 0 or side == 1:
                        bar_h = int(self.perturb_ratio * h)
                        new_img = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
                        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
                    else:
                        bar_w = int(self.perturb_ratio * w)
                        new_img = np.zeros((h, (w + bar_w), 3), dtype=np.uint8)
                        bar = np.zeros((h, bar_w, 3), dtype=np.uint8)
                    if side == 0:
                        new_img[:h, :, :] = img
                        new_img[h:, :, :] = bar
                    elif side == 1:
                        new_img[bar_h:, :, :] = img
                        new_img[:bar_h, :, :] = bar
                    elif side == 2:
                        new_img[:, :w, :] = img
                        new_img[:, w:, :] = bar
                    else:
                        new_img[:, bar_w:, :] = img
                        new_img[:, :bar_w, :] = bar
                    img = new_img
        return img


class Morphological_Ops(object):
    def __init__(self, do_mops=True):
        self.do_mops = do_mops

    def __call__(self, img):
        if self.do_mops:
            h, w, _ = img.shape
            # Erode or Dilate 25% of the times
            if random.random() <= 0.15:
                if w < 600 or h < 600:
                    iters = 1
                elif w < 1200 or h < 1200:
                    iters = 2
                elif w < 1800 or h < 1800:
                    iters = 3
                else:
                    iters = 4
                # Erode (65%) of the times
                if random.random() <= 0.65:
                    if w > 600 and h > 600:
                        kernel = np.ones((9, 9), dtype=np.uint8)
                        img = cv2.erode(img, kernel, iterations=iters - 1)
                # Dilate
                else:
                    if w > 600 and h > 600:
                        kernel = np.ones((4, 4), dtype=np.uint8)
                        img = cv2.dilate(img, kernel, iterations=iters)
        return img


def get_transforms(input_size, augment=True, rotate=True, debug=False):
    trans_list = []
    if augment and rotate:
        trans_list.append(
            SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        )
    trans_list.append(CropWhite(pad=5))

    # TODO: Add Dilation and Erosion
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.CropAndPad(percent=[-0.01, 0.00], keep_size=False, p=0.5),
            PadWhite(pad_ratio=0.4, p=0.2),
            A.Downscale(scale_min=0.2, scale_max=0.5, interpolation=3),
            A.Blur(),
            A.GaussNoise(),
            SaltAndPepperNoise(num_dots=20, p=0.5),
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(
        trans_list, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
    )


def add_functional_group(indigo, mol, debug=False):
    if random.random() > INDIGO_FUNCTIONAL_GROUP_PROB:
        return mol
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == "H":
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol


def add_explicit_hydrogen(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except Exception:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom("H")
            h.addBond(atom, 1)
    return mol


def add_rgroup(indigo, mol, smiles):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except Exception:
            continue
    if len(atoms) > 0 and "*" not in smiles:
        if random.random() < INDIGO_RGROUP_PROB:
            atom_idx = random.choice(range(len(atoms)))
            atom = atoms[atom_idx]
            atoms.pop(atom_idx)
            symbol = random.choice(RGROUP_SYMBOLS)
            r = mol.addAtom(symbol)
            r.addBond(atom, 1)
    return mol


def get_rand_symb():
    symb = random.choice(ELEMENTS)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_lowercase)
    if random.random() < 0.1:
        symb += random.choice(string.ascii_uppercase)
    if random.random() < 0.1:
        symb = f"({gen_rand_condensed()})"
    return symb


def get_rand_num():
    if random.random() < 0.9:
        if random.random() < 0.8:
            return ""
        else:
            return str(random.randint(2, 9))
    else:
        return "1" + str(random.randint(2, 9))


def gen_rand_condensed():
    tokens = []
    for i in range(5):
        if i >= 1 and random.random() < 0.8:
            break
        tokens.append(get_rand_symb())
        tokens.append(get_rand_num())
    return "".join(tokens)


def add_rand_condensed(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except Exception:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_CONDENSED_PROB:
        atom = random.choice(atoms)
        symbol = gen_rand_condensed()
        r = mol.addAtom(symbol)
        r.addBond(atom, 1)
    return mol


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if "*" in smiles:
        part_a, part_b = smiles.split(" ", maxsplit=1)
        part_b = re.search(r"\$.*\$", part_b).group(0)[1:-1]
        symbols = [t for t in part_b.split(";") if len(t) > 0]
        output = ""
        cnt = 0
        for i, c in enumerate(part_a):
            if c != "*":
                output += c
            else:
                output += f"[{symbols[cnt]}]"
                cnt += 1
        return mol, output
    else:
        if " " in smiles:
            # special cases with extension
            smiles = smiles.split(" ")[0]
        return mol, smiles


def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption(
            "render-comment", str(random.randint(1, 20)) + random.choice(string.ascii_letters)
        )
        indigo.setOption("render-comment-font-size", random.randint(40, 60))
        indigo.setOption("render-comment-alignment", random.choice([0, 0.5, 1]))
        indigo.setOption("render-comment-position", random.choice(["top", "bottom"]))
        indigo.setOption("render-comment-offset", random.randint(2, 30))


def add_color(indigo, mol):
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption("render-coloring", True)
    if random.random() < INDIGO_COLOR_PROB:
        indigo.setOption("render-base-color", random.choice(list(COLORS.values())))
    if random.random() < INDIGO_COLOR_PROB:
        if random.random() < 0.5:
            indigo.setOption("render-highlight-color-enabled", True)
            indigo.setOption("render-highlight-color", random.choice(list(COLORS.values())))
        if random.random() < 0.5:
            indigo.setOption("render-highlight-thickness-enabled", True)
        for atom in mol.iterateAtoms():
            if random.random() < 0.1:
                atom.highlight()
    return mol


def get_graph(mol, image, shuffle_nodes=False, pseudo_coords=False):
    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        if pseudo_coords:
            x, y, z = atom.xyz()
        else:
            x, y = atom.coords()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    if pseudo_coords:
        coords = normalize_nodes(np.array(coords))
        h, w, _ = image.shape
        coords[:, 0] = coords[:, 0] * w
        coords[:, 1] = coords[:, 1] * h
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {"coords": coords, "symbols": symbols, "edges": edges, "num_atoms": len(symbols)}
    return graph


def generate_indigo_image(
    smiles,
    mol_augment=True,
    default_option=False,
    shuffle_nodes=False,
    pseudo_coords=False,
    include_condensed=True,
    debug=False,
):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption("render-output-format", "png")
    indigo.setOption("render-background-color", "1,1,1")
    indigo.setOption("render-stereo-style", "none")
    indigo.setOption("render-label-mode", "hetero")
    indigo.setOption("render-font-family", "Arial")

    if not default_option:
        thickness = random.uniform(
            0.5, 2
        )  # limit the sum of the following two parameters to be smaller than 4
        indigo.setOption("render-relative-thickness", thickness)
        indigo.setOption("render-bond-length", random.uniform(10, 200))
        indigo.setOption("render-bond-line-width", random.uniform(1, 4 - thickness))
        if random.random() < 0.5:
            indigo.setOption(
                "render-font-family", random.choice(["Arial", "Times", "Courier", "Helvetica"])
            )
        indigo.setOption("render-label-mode", random.choice(["hetero", "terminal-hetero"]))
        if random.random() < 0.8:
            indigo.setOption("render-implicit-hydrogens-visible", False)
        if random.random() < 0.05:
            indigo.setOption("render-stereo-style", "old")
        else:
            indigo.setOption("render-stereo-style", "none")
        # if random.random() < 0.2:
        #     indigo.setOption('render-atom-ids-visible', True)

    try:
        mol = indigo.loadMolecule(smiles)
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            else:
                mol.aromatize()
            smiles = mol.canonicalSmiles()
            add_comment(indigo)
            mol = add_explicit_hydrogen(indigo, mol)
            mol = add_rgroup(indigo, mol, smiles)
            if include_condensed:
                mol = add_rand_condensed(indigo, mol)
            mol = add_functional_group(indigo, mol, debug)
            mol = add_color(indigo, mol)
            mol, smiles = generate_output_smiles(indigo, mol)

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph = get_graph(mol, img, shuffle_nodes, pseudo_coords)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = np.array([[[255.0, 255.0, 255.0]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success


class TrainDataset(Dataset):
    def __init__(
        self,
        args,
        df,
        tokenizer,
        split="train",
        dynamic_indigo=False,
        downscaling_factors=(4, 2, 2, 2),
    ):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.downscaling_factors = downscaling_factors
        self.create_maps = GenerateMaps(args.input_size, self.downscaling_factors)
        self.perturb = GenerateRealPerturbations(
            perturb=args.perturb, perturb_ratio=args.perturb_ratio
        )
        self.morph = Morphological_Ops(do_mops=args.do_mops)

        if "file_path" in df.columns:
            self.file_paths = df["file_path"].values
            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df["file_path"]]
        self.smiles = df["SMILES"].values if "SMILES" in df.columns else None
        self.formats = args.formats
        self.labelled = split == "train"
        if self.labelled:
            self.labels = {}
            for format_ in self.formats:
                if format_ in ["atomtok", "inchi"]:
                    field = FORMAT_INFO[format_]["name"]
                    if field in df.columns:
                        self.labels[format_] = df[field].values
        self.transform = get_transforms(args.input_size, augment=(self.labelled and args.augment))
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = dynamic_indigo and split == "train"
        if self.labelled and not dynamic_indigo and args.coords_file is not None:
            if args.coords_file == "aux_file":
                self.coords_df = df
                self.pseudo_coords = True
            else:
                self.coords_df = pd.read_csv(args.coords_file)
                self.pseudo_coords = False
        else:
            self.coords_df = None
            self.pseudo_coords = args.pseudo_coords

    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[], renormalize=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented["image"]
        if len(coords) > 0:
            coords = np.array(augmented["keypoints"])
            if renormalize:
                coords = normalize_nodes(coords, flip_y=False)
            else:
                _, height, width = image.shape
                coords[:, 0] = coords[:, 0] / width
                coords[:, 1] = coords[:, 1] / height
            coords = np.array(coords).clip(0, 1)
            return image, coords
        return image

    def __getitem__(self, idx):
        try:
            return self.gen_images(idx)
        except Exception as e:
            with open(
                os.path.join(self.args.save_path, f"error_dataset_{int(time.time())}.log"), "w"
            ) as f:
                f.write(str(e))
            raise e

    def gen_images(self, idx):  # noqa: C901
        ref = {}
        if self.dynamic_indigo:
            begin = time.time()
            image, smiles, graph, success = generate_indigo_image(
                self.smiles[idx],
                mol_augment=self.args.mol_augment,
                default_option=self.args.default_option,
                shuffle_nodes=self.args.shuffle_nodes,
                pseudo_coords=self.pseudo_coords,
                include_condensed=self.args.include_condensed,
            )

            # Generate Pixel Maps (Feature Addition)
            image = image.astype(np.uint8)
            pixel_maps = self.create_maps(image)

            # Additional Perturbations (Crop and Bar)
            image = self.perturb(image)

            end = time.time()
            if idx < 30 and self.args.save_image:
                path = os.path.join(self.args.save_path, "images")
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, f"{idx}.png"), image)
            if not success:
                return idx, None, {}

            # Morphological Operations
            image = self.morph(image)

            image, coords = self.image_transform(
                image, graph["coords"], renormalize=self.pseudo_coords
            )

            graph["coords"] = coords
            ref["time"] = end - begin
            if "atomtok" in self.formats:
                max_len = FORMAT_INFO["atomtok"]["max_len"]
                label = self.tokenizer["atomtok"].text_to_sequence(smiles, tokenized=False)
                ref["atomtok"] = torch.LongTensor(label[:max_len])
            if (
                "edges" in self.formats
                and "atomtok_coords" not in self.formats
                and "chartok_coords" not in self.formats
            ):
                ref["edges"] = torch.tensor(graph["edges"])
            if "atomtok_coords" in self.formats:
                self._process_atomtok_coords(
                    idx,
                    ref,
                    smiles,
                    graph["coords"],
                    graph["edges"],
                    mask_ratio=self.args.mask_ratio,
                )
            if "chartok_coords" in self.formats:
                self._process_chartok_coords(
                    idx,
                    ref,
                    smiles,
                    graph["coords"],
                    graph["edges"],
                    mask_ratio=self.args.mask_ratio,
                )
            return idx, image, ref, pixel_maps

        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)

            # Generate Pixel Maps (Feature Addition)
            pixel_maps = self.create_maps(image)

            # Additional Perturbations (Crop and Bar)
            image = self.perturb(image)

            if image is None:
                image = np.array([[[255.0, 255.0, 255.0]] * 10] * 10).astype(np.float32)
                print(file_path, "not found!")
            if self.coords_df is not None:
                h, w, _ = image.shape
                coords = np.array(eval(self.coords_df.loc[idx, "node_coords"]))
                if self.pseudo_coords:
                    coords = normalize_nodes(coords)
                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                image, coords = self.image_transform(image, coords, renormalize=self.pseudo_coords)

            else:
                image = self.image_transform(image)
                coords = None
            if self.labelled:
                smiles = self.smiles[idx]
                if "atomtok" in self.formats:
                    max_len = FORMAT_INFO["atomtok"]["max_len"]
                    label = self.tokenizer["atomtok"].text_to_sequence(smiles, False)
                    ref["atomtok"] = torch.LongTensor(label[:max_len])
                if "atomtok_coords" in self.formats:
                    if coords is not None:
                        self._process_atomtok_coords(idx, ref, smiles, coords, mask_ratio=0)
                    else:
                        self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
                if "chartok_coords" in self.formats:
                    if coords is not None:
                        self._process_chartok_coords(idx, ref, smiles, coords, mask_ratio=0)
                    else:
                        self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
            if self.args.predict_coords and (
                "atomtok_coords" in self.formats or "chartok_coords" in self.formats
            ):
                smiles = self.smiles[idx]
                if "atomtok_coords" in self.formats:
                    self._process_atomtok_coords(idx, ref, smiles, mask_ratio=1)
                if "chartok_coords" in self.formats:
                    self._process_chartok_coords(idx, ref, smiles, mask_ratio=1)
            return idx, image, ref, pixel_maps

    def _process_atomtok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = FORMAT_INFO["atomtok_coords"]["max_len"]
        tokenizer = self.tokenizer["atomtok_coords"]
        if smiles is None or not isinstance(smiles, str):
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref["atomtok_coords"] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref["atom_indices"] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref["coords"] = torch.tensor(coords)
            else:
                ref["coords"] = torch.ones(len(indices), 2) * -1.0
        if edges is not None:
            ref["edges"] = torch.tensor(edges)[: len(indices), : len(indices)]
        else:
            if "edges" in self.df.columns:
                edge_list = eval(self.df.loc[idx, "edges"])
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in edge_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                ref["edges"] = edges
            else:
                ref["edges"] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)

    def _process_chartok_coords(self, idx, ref, smiles, coords=None, edges=None, mask_ratio=0):
        max_len = FORMAT_INFO["chartok_coords"]["max_len"]
        tokenizer = self.tokenizer["chartok_coords"]
        if smiles is None or not isinstance(smiles, str):
            smiles = ""
        label, indices = tokenizer.smiles_to_sequence(smiles, coords, mask_ratio=mask_ratio)
        ref["chartok_coords"] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref["atom_indices"] = torch.LongTensor(indices)
        if tokenizer.continuous_coords:
            if coords is not None:
                ref["coords"] = torch.tensor(coords)
            else:
                ref["coords"] = torch.ones(len(indices), 2) * -1.0
        if edges is not None:
            ref["edges"] = torch.tensor(edges)[: len(indices), : len(indices)]
        else:
            if "edges" in self.df.columns:
                edge_list = eval(self.df.loc[idx, "edges"])
                n = len(indices)
                edges = torch.zeros((n, n), dtype=torch.long)
                for u, v, t in edge_list:
                    if u < n and v < n:
                        if t <= 4:
                            edges[u, v] = t
                            edges[v, u] = t
                        else:
                            edges[u, v] = t
                            edges[v, u] = 11 - t
                ref["edges"] = edges
            else:
                ref["edges"] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)


class AuxTrainDataset(Dataset):
    def __init__(self, args, train_df, aux_df, tokenizer):
        super().__init__()
        self.train_dataset = TrainDataset(
            args, train_df, tokenizer, dynamic_indigo=args.dynamic_indigo
        )
        self.aux_dataset = TrainDataset(args, aux_df, tokenizer, dynamic_indigo=False)

    def __len__(self):
        return len(self.train_dataset) + len(self.aux_dataset)

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            return self.aux_dataset[idx - len(self.train_dataset)]


def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1 - i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1 - i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def bms_collate(batch):
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [
        k
        for k in formats
        if k in ["atomtok", "inchi", "nodes", "atomtok_coords", "chartok_coords", "atom_indices"]
    ]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Time
    # if 'time' in formats:
    #     refs['time'] = [ex[2]['time'] for ex in batch]
    # Coords
    if "coords" in formats:
        refs["coords"] = pad_sequence(
            [ex[2]["coords"] for ex in batch], batch_first=True, padding_value=-1.0
        )
    # Edges
    if "edges" in formats:
        edges_list = [ex[2]["edges"] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs["edges"] = torch.stack(
            [
                F.pad(edges, (0, max_len - len(edges), 0, max_len - len(edges)), value=-100)
                for edges in edges_list
            ],
            dim=0,
        )
    # Collate the pixel maps per stage
    stages = [[] for i in range(4)]
    for ex in batch:
        maps = ex[3]
        for i, stage in enumerate(maps):
            stages[i].append(stage)

    maps_final = []
    for stage in stages:
        maps_final.append(torch.stack(stage).type(torch.FloatTensor))

    return ids, pad_images(imgs), refs, maps_final
