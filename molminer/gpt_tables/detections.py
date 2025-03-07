import base64
import glob
import logging
import os
import re
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray
from openai import OpenAI
from tqdm import tqdm

from molminer.pipeline.data_structures import PDF

HEAD_NEG_KEYS = [
    "isn't",
    "not",
    "explicitly",
    "top",
    "column 1",
    "unable",
    "cannot",
    "?",
    "ocr",
    "can't",
    "not",
]
TABID_NEG_KEYS = [
    "isn't",
    "not",
    "explicitly",
    "top",
    "column 1",
    "unable",
    "cannot",
    "?",
    "ocr",
    "can't",
    "not",
    "no",
]
IGNORE_HEADERS = ["compound", "comp.", "structure"]
REF_SYNONYMS_EXACT = [
    "#",
    "compound",
    "compounds",
    "compound i.d.",
    "compound id",
    "compound no.",
    "compound #",
    "compound num",
    "compound number",
    "cpd. no.",
    "cpd. #",
    "cpd. num",
    "cmpd. no.",
    "cmpd. #",
    "cmpd. num",
    "example compound",
    "example #",
    "example compound no.",
    "ex. compound",
    "ex. #",
    "ex. no.",
    "ex. compound no.",
]
REF_SYNONYMS_SUBST = [
    "example",
    "e.g.",
    "no.",
    "number",
    "num",
    "compound",
]


logger = logging.getLogger("molminer.gpt_tables.detections")


class ColumnRenamer:
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return f"{x}_{self.d[x]}"


def _find_header_sequences(headers: List[str]) -> List[List[str]]:
    """Identifies sequences of non-repeating headers."""
    found = []
    sequences = []
    for col in headers:
        if col not in found:
            found.append(col)
        else:
            sequences.append(found)
            found = [col]
    sequences.append(found)
    # # If last sequence contains only one element, fuse them
    # if len(sequences[-1]) == 1 and len(sequences) == 2:
    #     sequences = [s for seq in sequences for s in seq]
    return sequences


def _normalize_dataframe_columns(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """
    Adjusts DataFrame columns to match target columns count.
    If too many columns, drops excess columns.
    If too few columns, adds empty columns.
    """
    current_cols = len(df.columns)
    target_cols = len(target_columns)

    # If DataFrame has too many columns, drop the extra ones
    if current_cols > target_cols:
        return df.iloc[:, :target_cols]

    # If DataFrame has too few columns, add empty columns
    if current_cols < target_cols:
        diff = target_cols - current_cols
        # Create empty columns with NaN values
        empty_cols = pd.DataFrame(
            np.nan, index=df.index, columns=[f"empty_{i}" for i in range(diff)]
        )
        return pd.concat([df, empty_cols], axis=1)

    return df


def _process_repeating_headers(
    dfs: List[pd.DataFrame], headers: List[str], sequences: List[List[str]]
) -> List[pd.DataFrame]:
    """Processes DataFrames with repeating headers."""
    if not all(seq == sequences[0] for seq in sequences):
        logger.debug(
            "This table appears to have repeating headers, but we "
            "could not find the pattern. We are going to concatenate "
            f"the tables as-is. Sequences found: {sequences}"
        )
        for df in dfs:
            df.rename(columns=ColumnRenamer(), inplace=True)
        return dfs

    new_dfs = []
    seq_len = len(sequences[0])

    for df in dfs:
        new_subdfs = []
        for i in range(len(sequences)):
            seq_start = i * seq_len
            seq_end = seq_start + seq_len
            logger.debug(f"NEW START: {seq_start}; NEW END: {seq_end}")

            sub_df = df.iloc[:, seq_start:seq_end]
            sub_df.columns = headers[0:seq_len]
            new_subdfs.append(sub_df)

        new_df = pd.concat(new_subdfs, ignore_index=True)
        new_dfs.append(new_df)

    return new_dfs


def _standardize_headers(dfs: List[pd.DataFrame], headers: List[str]):
    """Standardizes headers across all DataFrames."""
    ref_col_pos = None
    for i, head in enumerate(headers):
        if any(syn == head.lower() for syn in REF_SYNONYMS_EXACT):
            ref_col_pos = i
            break
    # If still none, use more relaxed criteria
    if ref_col_pos is None:
        for i, head in enumerate(headers):
            if any(syn in head.lower() for syn in REF_SYNONYMS_SUBST):
                ref_col_pos = i
                break
    if ref_col_pos is not None:
        headers[ref_col_pos] = "REF"

    for df in dfs:
        df.columns = headers


def _concat_table(dfs: List[pd.DataFrame], has_header: int = 1):
    """
    Concatenate dataframes into 1 and keep the header only from the first element
    Change the ID of the references to a common header -- REF
    mode=1: Header Table mode=0: Non-header Table
    """
    if not has_header:
        return pd.concat(dfs, ignore_index=True)

    headers = dfs[0].columns.tolist()

    # Normalize column counts
    normalized_dfs = [dfs[0]]
    for df in dfs[1:]:
        normalized_df = _normalize_dataframe_columns(df, headers)
        normalized_dfs.append(normalized_df)
    dfs = normalized_dfs

    # Check for repeating headers, if found stack them vertically
    sequences = _find_header_sequences(headers)

    if len(sequences) > 1:
        logger.debug(f"Repeating headers found: {sequences}; attempting to resolve")
        dfs = _process_repeating_headers(dfs, headers, sequences)

    # Standardize headers across all DataFrames
    headers = dfs[0].columns.tolist()
    logger.debug("STANDARDIZING HEADERS....")
    _standardize_headers(dfs, headers)

    # Concatenate all tables (W or W/O headers)
    full_table = pd.concat(dfs, ignore_index=True)
    logger.debug("FULL TABLE RETURNING !!!")
    logger.debug(full_table)
    return full_table


def _compare_t_ids(last_t_id: str, table_id: str):
    logger.debug(f"LAST TABLE ID: {last_t_id}")
    logger.debug(f"CURRENT TABLE ID: {table_id}")
    # Split by words
    if " " in last_t_id:
        prev = last_t_id.split(" ")[1]
    else:
        prev = last_t_id
    if "-" in prev:
        prev = prev[0]
    prev = prev.split(".")[0]

    if " " in table_id:
        cur = table_id.split(" ")[1]
    else:
        cur = table_id
    if "-" in cur:
        cur = cur[0]
    cur = cur.split(".")[0]

    if prev == cur:
        logger.debug(f"In compare_t_ids, previd ({prev}) == curid {cur}")
        return True
    else:
        logger.debug("In compare_t_ids, previd != curid")
        return False


class TableAccumulator:
    def __init__(self):
        self.current_tables = []
        self.current_table_id = None
        self.table_counter = -1  # For tables without IDs

    def is_active(self) -> bool:
        return len(self.current_tables) > 0

    def start_new(self, table_df: pd.DataFrame, table_id: str | None):
        self.current_tables = [table_df]
        self.current_table_id = table_id

    def append(self, table_df: pd.DataFrame):
        self.current_tables.append(table_df)

    def should_append(self, new_table_id: str | None) -> bool:
        # If either table has no ID, assume it's a continuation
        if self.current_table_id is None or new_table_id is None:
            return True
        # Else compare the IDs
        return _compare_t_ids(self.current_table_id, new_table_id)

    def finalize(self, pdf_name: Optional[str] = None) -> dict:
        """Concatenate accumulated tables and return {id: final_table}"""
        if logger.level <= logging.DEBUG:
            logger.debug("FINALIZING TABLES")
            logger.debug(f"CURRENT TABLES: {len(self.current_tables[0])}")
            if pdf_name:
                pdf_name = pdf_name.split(".")[0]
                outdir = f"outputs/{pdf_name}/table_debugging/{self.current_table_id}"
            else:
                outdir = "outputs/table_debugging"
            os.makedirs(outdir, exist_ok=True)
            # Remove existing files before writing new ones
            # Assumption is that we're debugging b/c this is failing
            for f in glob.glob(os.path.join(outdir, "*.tsv")):
                os.remove(f)
            for i, table in enumerate(self.current_tables):
                with open(os.path.join(outdir, f"Table_{i}.tsv"), "w") as fh:
                    table.to_csv(fh, sep="\t")

        # logger.debug(f'CURRENT TABLES BEFORE CONCAT TABLE: {self.current_tables}')
        if not self.current_tables:
            return {}
        has_header = 1 if isinstance(self.current_tables[0].columns[0], str) else 0
        final_table = _concat_table(self.current_tables, has_header=has_header)

        # Generate table ID
        table_id = (
            self.current_table_id if self.current_table_id is not None else self.table_counter
        )
        if self.current_table_id is None:
            self.table_counter -= 1

        # Reset state
        self.current_tables = []
        self.current_table_id = None

        return {table_id: final_table}


class GPT_Table(object):
    def __init__(self, oai_api_key: str, model: str = "gpt-4o") -> None:
        self.client = OpenAI(api_key=oai_api_key)
        self.model = model

    def process_pdfs(self, pdfs: List[PDF]):
        for pdf in pdfs:
            logger.debug(f"Processing Table Stage for PDF: {pdf.name}")
            pdf.tables = self._extract_tables_from_pdf(pdf)
            logger.debug(f"PDF TABLES: \n\n{pdf.tables}")

    def _extract_tables_from_pdf(self, pdf: PDF) -> dict:
        """Extract tables from a PDF, handling multi-page tables."""
        tables = {}
        current_table = TableAccumulator()

        for page_num, page in tqdm(
            enumerate(pdf.pages), desc="Pages Processed: ", total=len(pdf.pages)
        ):
            table_df, table_id = self.process_page(page.page_img, page_num)

            if table_df is None:
                # No table on this page - save any accumulated table
                if current_table.is_active():
                    tables.update(current_table.finalize(pdf.name))
                continue

            if not current_table.is_active():
                # Starting a new table
                current_table.start_new(table_df, table_id)
            else:
                # Check if this is a continuation or a new table
                if current_table.should_append(table_id):
                    current_table.append(table_df)
                else:
                    # Save current table and start a new one
                    tables.update(current_table.finalize(pdf.name))
                    current_table.start_new(table_df, table_id)

        # Save any remaining table
        if current_table.is_active():
            tables.update(current_table.finalize(pdf.name))

        return tables

    def temp_viz(self, img: ndarray):
        cv2.imwrite("Temp_Viz.png", img)

    def process_page(self, page_img: ndarray, page_num: int):
        # self.temp_viz(page_img)
        _, buf = cv2.imencode(".png", page_img)
        b64_img = base64.b64encode(buf).decode("utf-8")

        # GPT-4o BLOCK
        try:
            # PROMPT 1: Detect if there is a Table in the Page
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Detect and Extract Tables from Document Images",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Is there a table in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                            },
                        ],
                    },
                ],
                temperature=0.0,
            )

            resp1 = response.choices[0].message.content
            logger.debug(f"In process_page, resp1: {resp1}")

            if resp1 is not None and "yes" in resp1.lower():
                # PROMPT 2: Detect if table has any useful metadata
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Detect and Extract Tables from Document Images",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Are there any columns in table excluding chemical "
                                    + "structures and their respective numbers?",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                                },
                            ],
                        },
                        # {"role": "assistant", "content": resp1}
                    ],
                    temperature=0.0,
                )

                resp2 = response.choices[0].message.content
                if resp2 is None:
                    return None, None
                logger.debug(f"In process_page, resp2: {resp2}")

                if resp2 is not None and "yes" in resp2.lower():
                    # PROMPT 3: Detect the ID of the table
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Detect and Extract Tables from Document Images",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Does the Table have a label "
                                        + "just above the table?",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                                    },
                                ],
                            },
                            # {"role": "assistant", "content": resp1},
                            # {"role": "assistant", "content": resp2}
                        ],
                        temperature=0.0,
                    )

                    resp3 = response.choices[0].message.content
                    if resp3 is None:
                        return None, None
                    logger.debug(f"In process_page, resp3: {resp3}")

                    # Now Extract all the headers in the form of a TSV response
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Detect and Extract Tables from Document Images",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Extract the header row of the table \
                            if there is any. Be careful not to split multi-line \
                            headers into separate columns. Show it in a TSV format.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                                    },
                                ],
                            },
                            # {"role": "assistant", "content": resp1},
                            # {"role": "assistant", "content": resp2},
                            # {"role": "assistant", "content": resp3}
                        ],
                        temperature=0.0,
                    )
                    resp4 = response.choices[0].message.content
                    if resp4 is None:
                        return None, None
                    logger.debug(f"In process_page, resp4: {resp4}")

                    # Now Extract all the metadata in the form of a XML response
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Detect and Extract Tables from Document Images",
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Annotate the table \
                            in a TSV format based on rows only excluding any headers.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                                    },
                                ],
                            },
                            # {"role": "assistant", "content": resp1},
                            # {"role": "assistant", "content": resp2},
                            # {"role": "assistant", "content": resp3},
                            # {"role": "assistant", "content": resp4}
                        ],
                        temperature=0.0,
                    )
                    resp5 = response.choices[0].message.content
                    if resp5 is None:
                        return None, None
                    logger.debug(f"In process_page, resp5: {resp5}")
                    table_vals, table_id = self.parse_table(resp3, resp4, resp5, page_num)
                else:
                    table_vals, table_id = None, None
            else:
                table_vals, table_id = None, None
        except:  # noqa
            table_vals, table_id = None, None
        return table_vals, table_id

    # Function that parses the TSV out from GPT response
    def parse_table(self, id_block: str, head_block: str, body_block: str, page_num: int):  # noqa
        patt = re.compile(r"^([Tt]able [0-9A-Za-z]+)")
        # Check if there is an ID extracted
        if any(key in id_block.lower() for key in TABID_NEG_KEYS):
            table_id = None
            invalid_t = False
        # Split the string to get the exact table ID
        else:
            if '"' in id_block:
                id_block = id_block.split('"')[1]
                id_block = id_block.split('"')[0]
            id_block = id_block.split(",")[0]
            # See if we can extract a "Table 1." or something similar
            matches = re.match(patt, id_block)
            if matches is not None and len(matches.groups()) > 0:
                table_id = matches[0]
            else:
                table_id = id_block
            # If we couldn't extract a simpler ID, we'll use the full extracted text
            if len(table_id) > 50:
                invalid_t = True
            else:
                invalid_t = False
        if not invalid_t:
            # Initialize the header block if present
            if any(key in head_block.lower() for key in HEAD_NEG_KEYS):
                no_head = True
            else:
                if "```tsv" in head_block:
                    head_block = head_block.split("```tsv\n")[1]
                    head_block = head_block.split("```")[0]
                if "```plaintext" in head_block:
                    head_block = head_block.split("```plaintext\n")[1]
                    head_block = head_block.split("```")[0]
                elif "```" in head_block:
                    head_block = head_block.split("```")[1]
                    head_block = head_block.split("```")[0]

                if "\\t" in head_block:
                    headers = head_block.split("\\t")
                else:
                    headers = head_block.split("\t")
                headers = [h.lstrip("\n").rstrip("\n") for h in headers]
                headers = [h.strip() for h in headers]
                logger.debug(f"In GPT_Table.parse_table, headers: {headers}")
                no_head = False

            # Strip off all the data before and after tildas
            # Separate into rows
            if "```" in body_block:
                no_body = False
                if "```tsv" in body_block:
                    body_block = body_block.split("```tsv\n")[1]
                    body_block = body_block.split("```")[0]
                elif "```plaintext" in body_block:
                    body_block = body_block.split("```plaintext\n")[1]
                    body_block = body_block.split("```")[0]
                elif "```" in body_block:
                    body_block = body_block.split("```")[1]
                    body_block = body_block.split("```")[0]
            elif any(key in head_block.lower() for key in HEAD_NEG_KEYS):
                no_body = True

            if not no_body:
                body_block_lst = body_block.split("\n")
                body_block_lst = [b for b in body_block_lst if len(b)]
                rows = []
                for block in body_block_lst:
                    data = block.split("\t")
                    data = [d.strip() for d in data]
                    rows.append(data)
            else:
                no_body = True

            if not no_head and not no_body:
                logger.debug(f"In parse_table, OLD Header columns: {len(headers)}")
            if not no_body:
                logger.debug(f"In parse_table, OLD Data columns: {len(rows[0])}")
                logger.debug(f"In parse_table, PARSED ROWS: \n {rows}")

            # Only join header columns if more than data rows
            if not no_head and not no_body:
                if len(headers) > len(rows[0]):
                    # First try removing the structure column
                    headers = [h for h in headers if h not in IGNORE_HEADERS]

                    # See if we have repeating headers
                    seqs = _find_header_sequences(headers)
                    if len(seqs) > 1 and all(seq == seqs[0] for seq in seqs):
                        logger.debug(f"In parse_table, repeating headers: {seqs}")
                        headers = seqs[0]

                    # If we still can't fix the headers, combine the last n rows
                    # Not great, but minimizes causing errors in the pipeline
                    if len(headers) > len(rows[0]):
                        diff = len(headers) - len(rows[0])
                        new_head = ""
                        for temp_head in headers[len(headers) - diff - 1 :]:
                            new_head += temp_head
                        # Delete the last n headers
                        del headers[len(headers) - diff - 1 :]
                        headers.append(new_head)
                # Account for cases where the repeating headers are not correctly parsed by GPT
                if len(headers) < len(rows[0]):
                    diff = len(rows[0]) - len(headers)
                    diff_rows = headers[-diff:]
                    headers.extend(diff_rows)

            if not no_head and not no_body:
                logger.debug(f"parse_table, NEW Header columns: {len(headers)}")
                logger.debug(f"parse_table, NEW HEADERS: {headers}")
            if not no_body:
                logger.debug(f"parse_table, NEW Data columns: {len(rows[0])}")
            else:
                logger.debug("parse_table, TABLE COULD NOT BE PARSED")

            # Add the rows
            if not no_head and not no_body:
                # # This throws an error if there are duplicate column names
                # df = pd.DataFrame(rows, columns=headers)
                df = pd.DataFrame(rows)
                df.columns = headers
            elif no_head and not no_body:
                df = pd.DataFrame(rows)
            else:
                df = None
        else:
            logger.debug("parse_table, INVALID TABLE ID DETECTED, LIKELY HALLUCINATED TABLE")
            df, table_id = None, None

        return df, table_id
