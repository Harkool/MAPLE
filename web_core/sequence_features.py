import math

import pandas as pd
import streamlit as st

from web_core.ui_text import AA_ORDER_56, RAW56_UNIQUE_COLUMNS


ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
HYDROPHOBIC_AA = set("AVILMFWY")
POLAR_AA = set("STNQC")
AROMATIC_AA = set("FWY")
SMALL_AA = set("AGSTCP")

KD_SCALE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

RESIDUE_MASS = {
    "A": 89.09,
    "C": 121.15,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}
WATER_MASS = 18.01528
CHARGE_APPROX = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}

HYDRO_56 = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
CHARGE_56 = {"R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0, "H": 0.5}
WEIGHT_56 = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121,
    "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}
HELIX_56 = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}


def clean_sequence(seq: str) -> str:
    import re

    if seq is None:
        return ""
    letters_only = re.findall(r"[A-Za-z]", str(seq))
    return "".join(letters_only).upper()


def validate_sequence(seq: str) -> tuple:
    cleaned = clean_sequence(seq)
    if not cleaned:
        return False, "empty_after_cleaning"
    invalid_chars = sorted(set(cleaned) - ALLOWED_AA)
    if invalid_chars:
        return False, "invalid_residues:" + ",".join(invalid_chars)
    return True, ""


def length_warning(seq: str) -> str:
    if len(seq) < 5:
        return "shorter_than_5"
    if len(seq) > 200:
        return "longer_than_200"
    return ""


@st.cache_data(show_spinner=False)
def compute_sequence_properties(sequence: str) -> dict:
    seq = clean_sequence(sequence)
    length = len(seq)
    if length == 0:
        return {
            "length": 0,
            "molecular_weight_approx": None,
            "net_charge_approx": None,
            "charge_density": None,
            "mean_hydrophobicity_kyte_doolittle": None,
            "fraction_positive": None,
            "fraction_negative": None,
            "fraction_hydrophobic": None,
            "fraction_polar": None,
            "fraction_aromatic": None,
            "fraction_small": None,
            "fraction_glycine": None,
            "fraction_proline": None,
        }

    known = [aa for aa in seq if aa in ALLOWED_AA]
    known_length = len(known)

    molecular_weight = (
        sum(RESIDUE_MASS[aa] for aa in known) - WATER_MASS * (known_length - 1)
        if known
        else None
    )
    net_charge = sum(CHARGE_APPROX.get(aa, 0.0) for aa in known) if known else None
    hydrophobicity = sum(KD_SCALE[aa] for aa in known) / known_length if known else None

    return {
        "length": length,
        "molecular_weight_approx": round(molecular_weight, 4) if molecular_weight is not None else None,
        "net_charge_approx": round(net_charge, 4) if net_charge is not None else None,
        "charge_density": round(net_charge / length, 6) if net_charge is not None and length > 0 else None,
        "mean_hydrophobicity_kyte_doolittle": round(hydrophobicity, 6) if hydrophobicity is not None else None,
        "fraction_positive": round(sum(aa in POSITIVE_AA for aa in seq) / length, 6),
        "fraction_negative": round(sum(aa in NEGATIVE_AA for aa in seq) / length, 6),
        "fraction_hydrophobic": round(sum(aa in HYDROPHOBIC_AA for aa in seq) / length, 6),
        "fraction_polar": round(sum(aa in POLAR_AA for aa in seq) / length, 6),
        "fraction_aromatic": round(sum(aa in AROMATIC_AA for aa in seq) / length, 6),
        "fraction_small": round(sum(aa in SMALL_AA for aa in seq) / length, 6),
        "fraction_glycine": round(sum(aa == "G" for aa in seq) / length, 6),
        "fraction_proline": round(sum(aa == "P" for aa in seq) / length, 6),
    }


@st.cache_data(show_spinner=False)
def compute_merged_raw56_properties(sequence: str) -> dict:
    seq = clean_sequence(sequence)
    if not seq:
        return {col: None for col in RAW56_UNIQUE_COLUMNS}

    n = len(seq)
    local_rows = []
    for i, aa in enumerate(seq):
        left = seq[i - 1] if i > 0 else ""
        right = seq[i + 1] if i + 1 < n else ""
        window = seq[max(0, i - 3) : min(n, i + 4)]
        window_len = max(len(window), 1)
        position = i / (n - 1) if n > 1 else 0.0
        local_rows.append(
            {
                "local_window_coverage_mean": len(window) / 7.0,
                "local_window_aromatic_fraction_mean": sum(ch in AROMATIC_AA for ch in window) / window_len,
                "local_window_charged_fraction_mean": sum(ch in set("RKDEH") for ch in window) / window_len,
                "local_window_hydrophobic_fraction_mean": sum(ch in set("AILMFPWV") for ch in window) / window_len,
                "local_window_polar_fraction_mean": sum(ch in set("NQSTYC") for ch in window) / window_len,
                "local_window_hydrophobicity_mean": sum(HYDRO_56.get(ch, 0.0) for ch in window) / window_len,
                "local_window_charge_mean": sum(CHARGE_56.get(ch, 0.0) for ch in window) / window_len,
                "local_window_helix_propensity_mean": sum(HELIX_56.get(ch, 1.0) for ch in window) / window_len,
                "center_is_charged_fraction": float(aa in set("RKDEH")),
                "left_same_as_center_fraction": float(bool(left) and left == aa),
                "right_same_as_center_fraction": float(bool(right) and right == aa),
                "left_neighbor_hydrophobic_fraction": float(bool(left) and left in set("AILMFPWV")),
                "right_neighbor_hydrophobic_fraction": float(bool(right) and right in set("AILMFPWV")),
                "terminal_position_fraction": float(i == 0 or i == n - 1),
                "relative_position_mean": position,
                "reverse_relative_position_mean": 1.0 - position if n > 1 else 0.0,
                "sin_pi_position_mean": math.sin(math.pi * position),
                "cos_pi_position_mean": math.cos(math.pi * position),
                "sin_2pi_position_mean": math.sin(2 * math.pi * position),
                "cos_2pi_position_mean": math.cos(2 * math.pi * position),
            }
        )

    local_df = pd.DataFrame(local_rows)
    summary = {f"aa_fraction_{aa}": round(sum(ch == aa for ch in seq) / n, 6) for aa in AA_ORDER_56}
    summary.update(
        {
            "raw56_mean_residue_hydrophobicity": round(sum(HYDRO_56.get(ch, 0.0) for ch in seq) / n, 6),
            "raw56_mean_residue_charge": round(sum(CHARGE_56.get(ch, 0.0) for ch in seq) / n, 6),
            "raw56_mean_residue_weight_norm": round(sum(WEIGHT_56.get(ch, 120.0) / 200.0 for ch in seq) / n, 6),
            "raw56_mean_residue_helix_propensity": round(sum(HELIX_56.get(ch, 1.0) for ch in seq) / n, 6),
            "raw56_global_fraction_charged": round(sum(ch in set("RKDEH") for ch in seq) / n, 6),
        }
    )
    summary.update({col: round(float(local_df[col].mean()), 6) for col in local_df.columns})
    return summary
