"""
Helper Script that takes in the PDF object and determines and links if any molecules with their
reference number has been found and the references match table references extracted using GPT-4
If references match, the molecules are linked with their corresponding table metadata
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

from molminer.pipeline.data_structures import PDF

logger = logging.getLogger("molminer.pipeline.link_final")


def get_reference_to_table_mapping(tables: Dict) -> Dict:
    """Creates a mapping of reference numbers to table IDs."""
    id2ref = defaultdict(list)
    for tid, df in tables.items():
        if "REF" in df.columns:
            refs = df["REF"].tolist()
            for ref in refs:
                id2ref[ref].append(tid)
    return id2ref


def get_row_metadata(tables: Dict, table_id: str, mol_ref: str) -> Dict:
    """Retrieves metadata for a specific reference from a table."""
    df = tables[table_id]
    row_val = df.loc[df["REF"] == mol_ref]
    return row_val.to_dict("records")[0]


def process_molecule_reference(mol_ref: str, id2ref: Dict, tables: Dict) -> Optional[List]:
    """Processes a single molecule reference and returns its metadata."""
    if mol_ref is None:
        return None

    if mol_ref not in id2ref:
        return [None]

    ref_meta = []
    for table_id in id2ref[mol_ref]:
        row_val = get_row_metadata(tables, table_id, mol_ref)
        ref_meta.append(row_val)
    return ref_meta


def process_page(page, id2ref: Dict, tables: Dict) -> List:
    """Processes a single page and returns metadata for all molecules."""
    page_metadata: list = []

    if page.ref_nos is None:
        return page_metadata

    for mol_ref in page.ref_nos:
        metadata = process_molecule_reference(mol_ref, id2ref, tables)
        page_metadata.append(metadata)

    return page_metadata


def link_tables(pdfs: List[PDF]):
    """Main function to link molecules with their table metadata."""
    for pdf in pdfs:
        tables = pdf.tables
        if tables is None:
            logger.debug(f"No tables found for PDF: {pdf.name}; will not link tables")
            continue

        id2ref = get_reference_to_table_mapping(tables)

        for page in pdf.pages:
            page_meta = process_page(page, id2ref, tables)
            if page_meta:
                page.metadata = page_meta
