"""
Data Structure Class Definitions for PDFs -> Pages -> Boxes -> SMILES -> Confidence

AUTHOR - Abhisek Dey (Summer Intern, 2024)
"""

from typing import List, Optional

import pandas as pd
from numpy import ndarray

# TSV_HEADER = "PDF\tPage\tMolPageID\tBBOXx1\tBBOXy1\tBBOXx2\tBBOXy2\tCan_SMILES\tConfidence\n"
TSV_HEADER = [
    "PDF",
    "REF",
    "Page",
    "MolPageID",
    "BBOXx1",
    "BBOXy1",
    "BBOXx2",
    "BBOXy2",
    "Can_SMILES",
    "Confidence",
]


# Class definition of Page of a PDF
class Page(object):
    page_img: Optional[ndarray]
    page_num: Optional[int]
    overlaid_img: Optional[ndarray]
    boxes: Optional[ndarray]
    box_images: Optional[List[ndarray]]
    smiles: Optional[List[str]]
    conf: Optional[List[float]]
    metadata: Optional[List[List[dict]]]
    orient: Optional[str]
    ref_nos: Optional[List[str]]

    def __init__(
        self,
        page_img=None,
        page_num=None,
        overlaid_img=None,
        boxes=None,
        box_images=None,
        smiles=None,
        conf=None,
        metadata=None,
        orient=None,
        ref_nos=None,
    ):
        self.page_img = page_img
        self.page_num = page_num
        self.overlaid_img = overlaid_img
        self.boxes = boxes
        self.box_images = box_images
        self.smiles = smiles
        self.conf = conf
        self.metadata = metadata
        self.orient = orient
        self.ref_nos = ref_nos

    def add_main_info(self, boxes, smiles, conf):
        self.boxes = boxes
        self.smiles = smiles
        self.conf = conf

    def add_meta_info(self, page_img, overlaid_img, box_images):
        self.page_img = page_img
        self.overlaid_img = overlaid_img
        self.box_images = box_images

    def molecules_to_df(self) -> pd.DataFrame:
        data = []
        if self.boxes is not None:
            for i, box in enumerate(self.boxes):
                ref_no = self.ref_nos[i] if self.ref_nos is not None else None
                w_box = [str(x) for x in box]
                w_page = str(self.page_num) if self.page_num is not None else "-"
                mol_id = str(i + 1)
                w_smi = self.smiles[i] if self.smiles is not None else None
                w_conf = str("{0:.2f}".format(self.conf[i])) if self.conf is not None else None
                data.append(["", ref_no, w_page, mol_id, *w_box, w_smi, w_conf])
            return pd.DataFrame(data, columns=TSV_HEADER)
        return pd.DataFrame()

    def get_metadata_cols(self):
        meta_cols = set()
        if self.metadata is not None:
            for mol_data in self.metadata:
                if mol_data is not None:
                    for val in mol_data:
                        [meta_cols.add(x) for x in val.keys()]
        return list(meta_cols)

    def metadata_to_df(self) -> Optional[pd.DataFrame]:
        data = []
        if self.metadata is not None:
            for mol_data in self.metadata:
                if mol_data is not None:
                    for row in mol_data:
                        if row is not None:
                            data.append(row)
            df = pd.DataFrame(data)
            # We're only interested in data that we
            # can link to a compound
            if "REF" not in df.columns:
                return pd.DataFrame()
            df.set_index(["REF"], inplace=True)
            df = df.groupby("REF").bfill().groupby("REF").first()
            df = df.reset_index(drop=False)
            df.dropna(subset=["REF"], inplace=True)
            return df
        return pd.DataFrame()


# Class definition for storing objects per PDF
class PDF(object):
    name: str
    pages: List[Page]
    tables: Optional[dict]

    def __init__(self, name: str):
        self.name = name
        self.pages = []
        self.tables = None

    # Every call appends a Page object to the PDF object
    def __call__(self, page: Page):
        page.page_num = len(self.pages) + 1
        self.pages.append(page)

    def build_molecule_df(self):
        data = []
        for page in self.pages:
            data.append(page.molecules_to_df())
        df = pd.concat(data, ignore_index=True)
        df["PDF"] = self.name
        return df

    def build_metadata_df(self):
        data = []
        for page in self.pages:
            data.append(page.metadata_to_df())
        return pd.concat(data, ignore_index=True)

    def build_df(self):
        mol_df = self.build_molecule_df()
        meta_df = self.build_metadata_df()
        if len(meta_df) > 0 and "REF" in meta_df.columns and "REF" in mol_df.columns:
            comb = mol_df.merge(meta_df[~meta_df.REF.isnull()], on="REF", how="left")
            comb.drop_duplicates(inplace=True)
            return comb
        return mol_df
