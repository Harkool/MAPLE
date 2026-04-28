import io
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


def _build_sequence_df(records: List[Dict[str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["sequence_id", "sequence"])
    if "sequence_id" not in df.columns:
        df["sequence_id"] = [f"seq_{i + 1:03d}" for i in range(len(df))]
    if "sequence" not in df.columns:
        df["sequence"] = ""
    df["sequence_id"] = df["sequence_id"].fillna("").astype(str)
    blank_ids = df["sequence_id"].str.strip() == ""
    if blank_ids.any():
        df.loc[blank_ids, "sequence_id"] = [f"seq_{i + 1:03d}" for i in df.index[blank_ids]]
    df["sequence"] = df["sequence"].fillna("").astype(str)
    return df[["sequence_id", "sequence"]]


@st.cache_data(show_spinner=False)
def parse_fasta_text(text: str) -> pd.DataFrame:
    text = text or ""
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame(columns=["sequence_id", "sequence"])

    records = []
    current_header = None
    current_seq_lines: List[str] = []

    for line in stripped.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith(">"):
            if current_header is not None or current_seq_lines:
                records.append(
                    {
                        "sequence_id": current_header or f"seq_{len(records) + 1:03d}",
                        "sequence": "".join(current_seq_lines),
                    }
                )
            current_header = raw[1:].strip() or f"seq_{len(records) + 1:03d}"
            current_seq_lines = []
        else:
            current_seq_lines.append(raw)

    if current_header is not None or current_seq_lines:
        records.append(
            {
                "sequence_id": current_header or f"seq_{len(records) + 1:03d}",
                "sequence": "".join(current_seq_lines),
            }
        )

    if not records:
        return pd.DataFrame(columns=["sequence_id", "sequence"])
    return _build_sequence_df(records)


@st.cache_data(show_spinner=False)
def parse_manual_input(text: str) -> pd.DataFrame:
    text = text or ""
    stripped = text.strip()
    if not stripped:
        return pd.DataFrame(columns=["sequence_id", "sequence"])

    if ">" in stripped:
        return parse_fasta_text(stripped)

    records = []
    for line in stripped.splitlines():
        raw = line.strip()
        if not raw:
            continue
        records.append({"sequence_id": f"seq_{len(records) + 1:03d}", "sequence": raw})
    return _build_sequence_df(records)


def _coerce_csv_dataframe(
    df: pd.DataFrame,
    sequence_col: Optional[str],
    likely_sequence_columns: List[str],
    likely_id_columns: List[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sequence_id", "sequence"])

    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    columns_lower = {col.lower(): col for col in working.columns}

    if sequence_col is None:
        for candidate in likely_sequence_columns:
            if candidate in columns_lower:
                sequence_col = columns_lower[candidate]
                break
    if sequence_col is None or sequence_col not in working.columns:
        raise KeyError("No usable sequence column found.")

    id_col = None
    for candidate in likely_id_columns:
        if candidate in columns_lower:
            id_col = columns_lower[candidate]
            break

    result = pd.DataFrame()
    if id_col:
        result["sequence_id"] = working[id_col].astype(str)
    else:
        result["sequence_id"] = [f"seq_{i + 1:03d}" for i in range(len(working))]
    result["sequence"] = working[sequence_col].astype(str)
    return _build_sequence_df(result.to_dict(orient="records"))


@st.cache_data(show_spinner=False)
def _parse_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def parse_csv_upload(
    uploaded_file,
    likely_sequence_columns: List[str],
    likely_id_columns: List[str],
) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["sequence_id", "sequence"])
    raw_df = _parse_csv_bytes(uploaded_file.getvalue())
    return _coerce_csv_dataframe(
        raw_df,
        sequence_col=None,
        likely_sequence_columns=likely_sequence_columns,
        likely_id_columns=likely_id_columns,
    )


def coerce_csv_dataframe(
    df: pd.DataFrame,
    sequence_col: Optional[str],
    likely_sequence_columns: List[str],
    likely_id_columns: List[str],
) -> pd.DataFrame:
    return _coerce_csv_dataframe(df, sequence_col, likely_sequence_columns, likely_id_columns)
