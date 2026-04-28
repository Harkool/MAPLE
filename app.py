import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import torch
except Exception:
    torch = None

APP_DIR = Path(__file__).resolve().parent


APP_TITLE = "MAPLE: AMP Prediction and Sequence Profiling"
APP_DESCRIPTION = (
    "MAPLE predicts antimicrobial peptide identity and 14 AMP-related functional activities from peptide "
    "sequences. This local interface also reports basic physicochemical descriptors to support "
    "selectivity-oriented interpretation. "
)
APP_SELECTIVITY_NOTE = (
    "MAPLE supports selectivity-oriented AMP prioritization by jointly considering antimicrobial activity, "
    "toxicity-related predictions, physicochemical descriptors, and motif-level interpretability."
)

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

# Approximate residue masses in Da. This is a simple descriptor, not an experimental measurement.
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

CHARGE_APPROX = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
AA_ORDER_56 = "ACDEFGHIKLMNPQRSTVWY"
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

FUNCTIONAL_LABELS = [
    "anti_mammalian_cells",
    "antibacterial",
    "antibiofilm",
    "anticancer",
    "antifungal",
    "antigram_negative",
    "antigram_positive",
    "antihiv",
    "antimrsa",
    "antioxidant",
    "antiparasitic",
    "antiviral",
    "cytotoxic",
    "hemolytic",
]

CHECKPOINT_LABEL_MAP = {
    "anti_mammalian_cells": "anti_mammalian_cells",
    "antibacterial": "antibacterial",
    "antibiofilm": "antibiofilm",
    "anticancer": "anticancer",
    "antifungal": "antifungal",
    "antigram_negative": "antigram-negative",
    "antigram_positive": "antigram-positive",
    "antihiv": "antihiv",
    "antimrsa": "antimrsa",
    "antioxidant": "antioxidant",
    "antiparasitic": "antiparasitic",
    "antiviral": "antiviral",
    "cytotoxic": "cytotoxic",
    "hemolytic": "hemolytic",
}

LIKELY_ID_COLUMNS = ["sequence_id", "id", "name", "peptide_id", "header"]
LIKELY_SEQUENCE_COLUMNS = ["sequence", "seq", "peptide_sequence", "peptide", "aa_sequence"]

DISPLAY_LABELS = {
    "amp": "AMP likelihood",
    "anti_mammalian_cells": "Mammalian cell activity risk",
    "antibacterial": "Antibacterial activity",
    "antibiofilm": "Anti-biofilm activity",
    "anticancer": "Anticancer activity",
    "antifungal": "Antifungal activity",
    "antigram_negative": "Anti-Gram-negative activity",
    "antigram_positive": "Anti-Gram-positive activity",
    "antihiv": "Anti-HIV activity",
    "antimrsa": "Anti-MRSA activity",
    "antioxidant": "Antioxidant activity",
    "antiparasitic": "Antiparasitic activity",
    "antiviral": "Antiviral activity",
    "cytotoxic": "Cytotoxicity risk",
    "hemolytic": "Hemolysis risk",
}

PRIORITY_GROUP_DISPLAY = {
    "invalid_sequence": "Invalid sequence",
    "properties_only": "Properties only",
    "high_priority_selective": "High-priority selective candidate",
    "effective_but_toxicity_flagged": "Effective but toxicity-flagged",
    "low_antibacterial_potential": "Low antibacterial potential",
    "intermediate_or_uncertain": "Intermediate or uncertain",
    "unknown": "Unknown",
}

INVALID_REASON_DISPLAY = {
    "": "",
    "empty_after_cleaning": "Sequence became empty after cleaning",
}

LENGTH_WARNING_DISPLAY = {
    "": "",
    "shorter_than_5": "Very short sequence (< 5 aa)",
    "longer_than_200": "Very long sequence (> 200 aa)",
}

RAW56_UNIQUE_COLUMNS = (
    [f"aa_fraction_{aa}" for aa in AA_ORDER_56]
    + [
        "raw56_mean_residue_hydrophobicity",
        "raw56_mean_residue_charge",
        "raw56_mean_residue_weight_norm",
        "raw56_mean_residue_helix_propensity",
        "local_window_coverage_mean",
        "local_window_aromatic_fraction_mean",
        "local_window_charged_fraction_mean",
        "local_window_hydrophobic_fraction_mean",
        "local_window_polar_fraction_mean",
        "local_window_hydrophobicity_mean",
        "local_window_charge_mean",
        "local_window_helix_propensity_mean",
        "center_is_charged_fraction",
        "left_same_as_center_fraction",
        "right_same_as_center_fraction",
        "left_neighbor_hydrophobic_fraction",
        "right_neighbor_hydrophobic_fraction",
        "terminal_position_fraction",
        "relative_position_mean",
        "reverse_relative_position_mean",
        "sin_pi_position_mean",
        "cos_pi_position_mean",
        "sin_2pi_position_mean",
        "cos_2pi_position_mean",
        "raw56_global_fraction_charged",
    ]
)


def _normalize_label_key(label: str) -> str:
    label = str(label).strip()
    label = label.replace("-", "_")
    lower = label.lower()
    if lower == "amp":
        return "amp"
    return lower


def _pretty_label_name(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace("_", " ").title())


def _pretty_probability_name(label: str) -> str:
    return f"Predicted probability: {_pretty_label_name(label)}"


def _pretty_binary_name(label: str) -> str:
    return f"Above decision threshold: {_pretty_label_name(label)}"


def _display_name_for_column(column: str) -> str:
    if column == "P_AMP":
        return _pretty_probability_name("amp")
    if column == "AMP_label":
        return _pretty_binary_name("amp")
    if column.startswith("P_"):
        return _pretty_probability_name(column[2:])
    if column.endswith("_label"):
        return _pretty_binary_name(column[:-6])
    pretty_map = {
        "sequence_id": "Sequence ID",
        "sequence": "Sequence",
        "clean_sequence": "Sequence used for prediction",
        "valid": "Usable for prediction",
        "invalid_reason": "Why prediction was skipped",
        "length_warning": "Length note",
        "selectivity_score": "Selectivity score (antibacterial - hemolysis)",
        "priority_group": "Recommendation group",
        "n_antibacterial_selective_motifs": "No. of antibacterial-selective motifs",
        "n_hemolytic_associated_motifs": "No. of hemolysis-associated motifs",
        "n_dual_activity_motifs": "No. of dual-activity motifs",
        "motif_balance_score": "Motif balance score",
        "top_matched_motifs": "Top matched motifs",
        "interpretation": "Interpretation",
        "length": "Length (aa)",
        "molecular_weight_approx": "Approx. molecular weight",
        "net_charge_approx": "Approx. net charge",
        "charge_density": "Charge density",
        "mean_hydrophobicity_kyte_doolittle": "Mean hydrophobicity",
    }
    return pretty_map.get(column, column.replace("_", " ").strip().title())


def _pretty_invalid_reason(reason: str) -> str:
    if reason in INVALID_REASON_DISPLAY:
        return INVALID_REASON_DISPLAY[reason]
    if reason.startswith("invalid_residues:"):
        residues = reason.split(":", 1)[1]
        return f"Contains unsupported amino-acid letters: {residues}"
    return reason


def _pretty_length_warning(value: str) -> str:
    return LENGTH_WARNING_DISPLAY.get(value, value)


def _pretty_priority_group(value: str) -> str:
    if value in PRIORITY_GROUP_DISPLAY:
        return PRIORITY_GROUP_DISPLAY[value]
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def _chart_priority_group(value: str) -> str:
    chart_labels = {
        "High-priority selective candidate": "High-priority\nselective",
        "Effective but toxicity-flagged": "Active but\nflagged for toxicity",
        "Low antibacterial potential": "Low\nantibacterial\npotential",
        "Intermediate or uncertain": "Intermediate\nor uncertain",
        "Properties only": "Descriptors\nonly",
        "Invalid sequence": "Invalid\nsequence",
        "Unknown": "Unknown",
    }
    return chart_labels.get(str(value), str(value).replace(" ", "\n"))


def _wrap_chart_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace(" but ", "<br>but ").replace(" or ", "<br>or ").replace(" and ", "<br>and ").replace("-flagged", "<br>flagged")


def _pretty_yes_no_na(value) -> str:
    if pd.isna(value):
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    try:
        numeric = int(value)
        if numeric == 1:
            return "Yes"
        if numeric == 0:
            return "No"
    except Exception:
        pass
    return str(value)


def _pretty_threshold_source(source: str) -> str:
    if not source or source == "default_0.5":
        return "Default thresholds (0.5)"
    prefix = "threshold_file:"
    if source.startswith(prefix):
        return "Selected threshold file"
    return str(source)


@st.cache_data(show_spinner=False)
def _load_motif_reference() -> Tuple[Optional[pd.DataFrame], str]:
    """Load optional motif interpretation reference if available."""
    motif_path = APP_DIR / "Data" / "motif_reference.csv"
    if not motif_path.exists():
        return None, "Motif reference file not found. Motif-level interpretation is skipped."
    motif_df = pd.read_csv(motif_path)
    required = {"motif", "motif_type", "log2FC_antibacterial", "log2FC_hemolytic"}
    if not required.issubset(set(motif_df.columns)):
        return None, "Motif reference file is missing required columns. Motif-level interpretation is skipped."
    motif_df = motif_df.copy()
    motif_df["motif"] = motif_df["motif"].astype(str).str.upper().str.strip()
    motif_df["motif_type"] = motif_df["motif_type"].astype(str).str.strip()
    motif_df = motif_df[motif_df["motif"].str.len() > 0].drop_duplicates(subset=["motif", "motif_type"])
    return motif_df, "Motif reference file loaded successfully."


def _scan_sequence_motifs(sequence: str, motif_df: Optional[pd.DataFrame]) -> Dict[str, object]:
    """Scan overlapping 7-mers against the motif reference."""
    default = {
        "n_antibacterial_selective_motifs": 0,
        "n_hemolytic_associated_motifs": 0,
        "n_dual_activity_motifs": 0,
        "motif_balance_score": 0,
        "top_matched_motifs": "",
    }
    if motif_df is None:
        return default

    seq = clean_sequence(sequence)
    if len(seq) < 7:
        return default

    motif_lookup = {}
    for _, row in motif_df.iterrows():
        motif_lookup.setdefault(row["motif"], []).append(row)

    matched_rows = []
    seen = set()
    for i in range(len(seq) - 6):
        kmer = seq[i : i + 7]
        if kmer in motif_lookup:
            for row in motif_lookup[kmer]:
                key = (kmer, row["motif_type"])
                if key not in seen:
                    seen.add(key)
                    matched_rows.append(row)

    if not matched_rows:
        return default

    matched_df = pd.DataFrame(matched_rows)
    n_antibacterial = int((matched_df["motif_type"] == "antibacterial_selective").sum())
    n_hemolytic = int((matched_df["motif_type"] == "hemolytic_associated").sum())
    n_dual = int((matched_df["motif_type"] == "dual_activity").sum())

    display_rows = []
    for _, row in matched_df.head(5).iterrows():
        display_rows.append(f"{row['motif']} ({row['motif_type']})")

    return {
        "n_antibacterial_selective_motifs": n_antibacterial,
        "n_hemolytic_associated_motifs": n_hemolytic,
        "n_dual_activity_motifs": n_dual,
        "motif_balance_score": n_antibacterial - n_hemolytic,
        "top_matched_motifs": "; ".join(display_rows),
    }


def _generate_interpretation(row, model_available: bool) -> str:
    """Generate a short candidate interpretation."""
    if not bool(row.get("valid", False)):
        reason = row.get("invalid_reason", "")
        return "Invalid sequence: prediction was skipped." if not reason else f"Invalid sequence: { _pretty_invalid_reason(reason) }."

    if not model_available or pd.isna(row.get("P_AMP")):
        return "Descriptor-only result: model checkpoints were not loaded, so only sequence descriptors and motif matching are shown."

    group = row.get("priority_group")
    motif_balance = row.get("motif_balance_score")
    motif_hint = ""
    if motif_balance is not None and not pd.isna(motif_balance):
        if motif_balance > 0:
            motif_hint = " and more antibacterial-selective than hemolysis-associated motif matches."
        elif motif_balance < 0:
            motif_hint = " and more hemolysis-associated than antibacterial-selective motif matches."
        else:
            motif_hint = " and a balanced motif match profile."

    if group == "high_priority_selective":
        return (
            "High-priority selective candidate: high predicted antibacterial activity, low predicted hemolysis, "
            "and a favorable selectivity profile" + motif_hint
        )
    if group == "effective_but_toxicity_flagged":
        return "Toxicity-flagged candidate: predicted antibacterial activity is accompanied by elevated hemolytic or cytotoxic probability."
    if group == "low_antibacterial_potential":
        return "Low antibacterial-potential candidate: AMP-related activity was not supported strongly enough for prioritization."
    return "Intermediate candidate: some activity signals are present, but the efficacy-to-toxicity profile is not clearly selective."


def _build_display_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
    """Create a user-facing dataframe with readable column names and values."""
    display_df = final_df.copy()

    if "valid" in display_df.columns:
        display_df["valid"] = display_df["valid"].map(_pretty_yes_no_na)
    if "AMP_label" in display_df.columns:
        display_df["AMP_label"] = display_df["AMP_label"].map(_pretty_yes_no_na)
    for label in FUNCTIONAL_LABELS:
        label_col = f"{label}_label"
        if label_col in display_df.columns:
            display_df[label_col] = display_df[label_col].map(_pretty_yes_no_na)

    if "invalid_reason" in display_df.columns:
        display_df["invalid_reason"] = display_df["invalid_reason"].fillna("").map(_pretty_invalid_reason)
    if "length_warning" in display_df.columns:
        display_df["length_warning"] = display_df["length_warning"].fillna("").map(_pretty_length_warning)
    if "priority_group" in display_df.columns:
        display_df["priority_group"] = display_df["priority_group"].fillna("unknown").map(_pretty_priority_group)

    rename_map = {
        "sequence_id": "Sequence ID",
        "sequence": "Sequence",
        "clean_sequence": "Sequence used for prediction",
        "valid": "Usable for prediction",
        "invalid_reason": "Why prediction was skipped",
        "length_warning": "Length note",
        "P_AMP": _pretty_probability_name("amp"),
        "AMP_label": _pretty_binary_name("amp"),
        "selectivity_score": "Selectivity score (antibacterial - hemolysis)",
        "priority_group": "Recommendation group",
        "n_antibacterial_selective_motifs": "No. of antibacterial-selective motifs",
        "n_hemolytic_associated_motifs": "No. of hemolysis-associated motifs",
        "n_dual_activity_motifs": "No. of dual-activity motifs",
        "motif_balance_score": "Motif balance score",
        "top_matched_motifs": "Top matched motifs",
        "interpretation": "Interpretation",
        "length": "Length (aa)",
        "molecular_weight_approx": "Approx. molecular weight",
        "net_charge_approx": "Approx. net charge",
        "charge_density": "Charge density",
        "mean_hydrophobicity_kyte_doolittle": "Mean hydrophobicity",
        "fraction_positive": "Fraction of positively charged residues",
        "fraction_negative": "Fraction of negatively charged residues",
        "fraction_hydrophobic": "Fraction of hydrophobic residues",
        "fraction_polar": "Fraction of polar residues",
        "fraction_aromatic": "Fraction of aromatic residues",
        "fraction_small": "Fraction of small residues",
        "fraction_glycine": "Fraction of glycine",
        "fraction_proline": "Fraction of proline",
        "raw56_mean_residue_hydrophobicity": "Average residue hydrophobicity (descriptor set)",
        "raw56_mean_residue_charge": "Average residue charge (descriptor set)",
        "raw56_mean_residue_weight_norm": "Average residue weight, normalized (descriptor set)",
        "raw56_mean_residue_helix_propensity": "Average helix propensity (descriptor set)",
        "local_window_coverage_mean": "Average local-window coverage",
        "local_window_aromatic_fraction_mean": "Average local aromatic-residue fraction",
        "local_window_charged_fraction_mean": "Average local charged-residue fraction",
        "local_window_hydrophobic_fraction_mean": "Average local hydrophobic-residue fraction",
        "local_window_polar_fraction_mean": "Average local polar-residue fraction",
        "local_window_hydrophobicity_mean": "Average local hydrophobicity",
        "local_window_charge_mean": "Average local charge",
        "local_window_helix_propensity_mean": "Average local helix propensity",
        "center_is_charged_fraction": "Charged-center frequency",
        "left_same_as_center_fraction": "Left-neighbor identity match frequency",
        "right_same_as_center_fraction": "Right-neighbor identity match frequency",
        "left_neighbor_hydrophobic_fraction": "Left-neighbor hydrophobic frequency",
        "right_neighbor_hydrophobic_fraction": "Right-neighbor hydrophobic frequency",
        "terminal_position_fraction": "Terminal-position frequency",
        "relative_position_mean": "Average relative position",
        "reverse_relative_position_mean": "Average reverse relative position",
        "sin_pi_position_mean": "Average sinusoidal position signal (pi)",
        "cos_pi_position_mean": "Average cosine position signal (pi)",
        "sin_2pi_position_mean": "Average sinusoidal position signal (2pi)",
        "cos_2pi_position_mean": "Average cosine position signal (2pi)",
        "raw56_global_fraction_charged": "Overall charged-residue fraction (descriptor set)",
    }

    for label in FUNCTIONAL_LABELS:
        rename_map[f"P_{label}"] = _pretty_probability_name(label)
        rename_map[f"{label}_label"] = _pretty_binary_name(label)
    for aa in AA_ORDER_56:
        rename_map[f"aa_fraction_{aa}"] = f"Amino-acid fraction: {aa}"

    return display_df.rename(columns=rename_map)


def _build_summary_text(final_df: pd.DataFrame, model_available: bool) -> str:
    """Generate a short plain-language summary for non-technical users."""
    total = len(final_df)
    valid_count = int(final_df["valid"].fillna(False).sum()) if "valid" in final_df.columns else 0
    invalid_count = total - valid_count
    high_priority = int((final_df.get("priority_group") == "high_priority_selective").sum()) if "priority_group" in final_df.columns else 0
    possible_toxic = int((final_df.get("priority_group") == "effective_but_toxicity_flagged").sum()) if "priority_group" in final_df.columns else 0
    predicted_amp = int((final_df.get("AMP_label").fillna(0) == 1).sum()) if "AMP_label" in final_df.columns and model_available else 0

    if total == 0:
        return "No sequences were available for analysis."

    if not model_available:
        return (
            f"{total} sequences were processed. {valid_count} could be profiled and {invalid_count} were skipped "
            f"for prediction-related analysis. Model predictions are currently unavailable, so the table shows "
            f"physicochemical descriptors and additional sequence descriptors only."
        )

    return (
        f"{total} sequences were submitted. {valid_count} were usable for prediction and {invalid_count} were skipped. "
        f"{predicted_amp} sequence(s) passed the AMP screen, {high_priority} were flagged as high-priority candidates, "
        f"and {possible_toxic} showed possible toxicity concerns."
    )


def _select_display_columns(display_df: pd.DataFrame) -> List[str]:
    """Choose a compact default column set for the main results table."""
    preferred = [
        "Sequence ID",
        "Sequence",
        "Sequence used for prediction",
        "Usable for prediction",
        "Why prediction was skipped",
        "Length note",
        "Predicted probability: AMP likelihood",
        "Above decision threshold: AMP likelihood",
        "Predicted probability: Antibacterial activity",
        "Predicted probability: Hemolysis risk",
        "Predicted probability: Cytotoxicity risk",
        "Predicted probability: Mammalian cell activity risk",
        "Selectivity score (antibacterial - hemolysis)",
        "Recommendation group",
        "Interpretation",
        "Motif balance score",
        "Top matched motifs",
        "Length (aa)",
        "Approx. net charge",
        "Mean hydrophobicity",
    ]
    return [col for col in preferred if col in display_df.columns]


def _select_raw56_display_columns(display_df: pd.DataFrame) -> List[str]:
    preferred = ["Sequence ID", "Sequence", "Sequence used for prediction"]
    preferred.extend([f"Amino-acid fraction: {aa}" for aa in AA_ORDER_56])
    preferred.extend(
        [
            "Average residue hydrophobicity (descriptor set)",
            "Average residue charge (descriptor set)",
            "Average residue weight, normalized (descriptor set)",
            "Average helix propensity (descriptor set)",
            "Average local-window coverage",
            "Average local aromatic-residue fraction",
            "Average local charged-residue fraction",
            "Average local hydrophobic-residue fraction",
            "Average local polar-residue fraction",
            "Average local hydrophobicity",
            "Average local charge",
            "Average local helix propensity",
            "Charged-center frequency",
            "Left-neighbor identity match frequency",
            "Right-neighbor identity match frequency",
            "Left-neighbor hydrophobic frequency",
            "Right-neighbor hydrophobic frequency",
            "Terminal-position frequency",
            "Average relative position",
            "Average reverse relative position",
            "Average sinusoidal position signal (pi)",
            "Average cosine position signal (pi)",
            "Average sinusoidal position signal (2pi)",
            "Average cosine position signal (2pi)",
            "Overall charged-residue fraction (descriptor set)",
        ]
    )
    return [col for col in preferred if col in display_df.columns]


def _default_checkpoint_dir() -> str:
    return "MAPLE_checkpoints"


def _default_threshold_file() -> str:
    checkpoint_dir = _default_checkpoint_dir()
    return f"{checkpoint_dir}/thresholds.json"


def _resolve_local_path(path_str: str) -> Path:
    """Resolve local files relative to the app directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return APP_DIR / path


def _candidate_checkpoint_dirs(checkpoint_dir: str) -> List[Path]:
    requested = _resolve_local_path(checkpoint_dir)
    candidates = [requested]
    if requested.name == "MAPLE_checkpoints":
        candidates.extend([APP_DIR / "Model", APP_DIR / "checkpoints"])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _candidate_threshold_files(threshold_file: str) -> List[Path]:
    requested = _resolve_local_path(threshold_file)
    candidates = [requested]
    if requested.name == "thresholds.json" and requested.parent.name == "MAPLE_checkpoints":
        candidates.extend([APP_DIR / "Model" / "thresholds.json", APP_DIR / "checkpoints" / "thresholds.json"])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def clean_sequence(seq: str) -> str:
    """Normalize a peptide sequence by keeping letters only and uppercasing."""
    if seq is None:
        return ""
    letters_only = re.findall(r"[A-Za-z]", str(seq))
    return "".join(letters_only).upper()


def validate_sequence(seq: str) -> tuple:
    """Validate a cleaned sequence against the 20 standard amino acids."""
    cleaned = clean_sequence(seq)
    if not cleaned:
        return False, "empty_after_cleaning"
    invalid_chars = sorted(set(cleaned) - ALLOWED_AA)
    if invalid_chars:
        return False, "invalid_residues:" + ",".join(invalid_chars)
    return True, ""


def _length_warning(seq: str) -> str:
    if len(seq) < 5:
        return "shorter_than_5"
    if len(seq) > 200:
        return "longer_than_200"
    return ""


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
def parse_manual_input(text: str) -> pd.DataFrame:
    """Parse manual textarea input into sequence_id and sequence columns."""
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


def _coerce_csv_dataframe(df: pd.DataFrame, sequence_col: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["sequence_id", "sequence"])

    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    columns_lower = {col.lower(): col for col in working.columns}

    if sequence_col is None:
        for candidate in LIKELY_SEQUENCE_COLUMNS:
            if candidate in columns_lower:
                sequence_col = columns_lower[candidate]
                break
    if sequence_col is None or sequence_col not in working.columns:
        raise KeyError("No usable sequence column found.")

    id_col = None
    for candidate in LIKELY_ID_COLUMNS:
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


def parse_csv_upload(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded CSV and auto-detect sequence and ID columns when possible."""
    if uploaded_file is None:
        return pd.DataFrame(columns=["sequence_id", "sequence"])
    raw_df = _parse_csv_bytes(uploaded_file.getvalue())
    return _coerce_csv_dataframe(raw_df)


@st.cache_data(show_spinner=False)
def parse_fasta_text(text: str) -> pd.DataFrame:
    """Parse FASTA text into sequence_id and sequence columns."""
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
def compute_sequence_properties(sequence: str) -> dict:
    """Compute basic approximate physicochemical descriptors for one sequence."""
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

    molecular_weight = sum(RESIDUE_MASS[aa] for aa in known) if known else None
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
    """Compute a readable, de-duplicated summary of the raw 56-d knowledge descriptors."""
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


def load_thresholds(threshold_file: str) -> tuple:
    """Load per-task thresholds, falling back to 0.5 defaults."""
    thresholds = {"amp": 0.5}
    thresholds.update({label: 0.5 for label in FUNCTIONAL_LABELS})
    source = "default_0.5"

    resolved_threshold_file = None
    threshold_candidates = _candidate_threshold_files(threshold_file) if threshold_file else []
    for candidate in threshold_candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            for key, value in loaded.items():
                normalized = _normalize_label_key(key)
                if normalized in thresholds:
                    try:
                        thresholds[normalized] = float(value)
                    except Exception:
                        continue
            resolved_threshold_file = candidate
            source = f"threshold_file:{resolved_threshold_file}"
            break
        except Exception:
            source = "default_0.5"

    return thresholds, source


def resolve_device(device_choice: str) -> str:
    """Resolve auto/cpu/cuda into an executable device string."""
    choice = (device_choice or "auto").lower()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if choice == "cuda":
        return "cuda" if cuda_available else "cpu"
    if choice == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def _find_amp_checkpoint(checkpoint_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(checkpoint_dir, "AMP.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _find_label_checkpoint(checkpoint_dir: str, label: str) -> Optional[str]:
    basename = CHECKPOINT_LABEL_MAP[label]
    candidates = [
        os.path.join(checkpoint_dir, "label", f"{basename}.pt"),
        os.path.join(checkpoint_dir, "label", basename, f"{basename}.pt"),
        os.path.join(checkpoint_dir, f"{basename}.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def _load_checkpoint_model(checkpoint_path: str, device: str):
    import torch as _torch

    from model import MAPLE, safe_load_checkpoint

    checkpoint = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    num_labels = len(ckpt_args.get("label_cols", ["label"]))
    model = MAPLE(
        linsize=ckpt_args.get("hidden_size", 1024),
        lindropout=ckpt_args.get("dropout", 0.8),
        num_labels=num_labels,
        esm_dim=checkpoint.get("esm_dim", 480) if isinstance(checkpoint, dict) else 480,
        knowledge_dim=checkpoint.get("knowledge_dim", 256) if isinstance(checkpoint, dict) else 256,
    )
    model = safe_load_checkpoint(model, checkpoint_path, device=device)
    model = model.to(device)
    model.eval()
    return {"model": model, "checkpoint": checkpoint, "num_labels": num_labels}


@st.cache_resource(show_spinner=False)
def try_load_maple_predictor(checkpoint_dir: str, device: str):
    """Try to build a MAPLE predictor adapter without crashing the UI."""
    resolved_checkpoint_dir = str(_candidate_checkpoint_dirs(checkpoint_dir)[0])
    predictor = {
        "model_available": False,
        "warning": "",
        "checkpoint_dir": resolved_checkpoint_dir,
        "device": device,
        "amp_checkpoint": None,
        "label_checkpoints": {},
        "knowledge_transformer_ckpt": None,
        "build_unified_pkl": None,
        "QuadOutputDataset": None,
        "quad_output_collate_fn": None,
        "torch": None,
        "predict_module_reused": False,
    }

    try:
        from Generate_pkl import build_unified_pkl
        from data import QuadOutputDataset, quad_output_collate_fn

        predictor["build_unified_pkl"] = build_unified_pkl
        predictor["QuadOutputDataset"] = QuadOutputDataset
        predictor["quad_output_collate_fn"] = quad_output_collate_fn
        predictor["torch"] = __import__("torch")

        try:
            import predict as _predict_module

            predictor["predict_module_reused"] = True
            predictor["predict_module_name"] = _predict_module.__name__
        except Exception:
            predictor["predict_module_reused"] = False

        missing_report = None
        for candidate_dir in _candidate_checkpoint_dirs(checkpoint_dir):
            candidate_dir_str = str(candidate_dir)
            predictor["checkpoint_dir"] = candidate_dir_str
            predictor["knowledge_transformer_ckpt"] = os.path.join(candidate_dir_str, "knowledge_transformer.pt")
            predictor["amp_checkpoint"] = _find_amp_checkpoint(candidate_dir_str)
            predictor["label_checkpoints"] = {}
            if predictor["amp_checkpoint"] is None:
                missing_report = "AMP checkpoint was not found."
                continue

            missing = []
            for label in FUNCTIONAL_LABELS:
                path = _find_label_checkpoint(candidate_dir_str, label)
                predictor["label_checkpoints"][label] = path
                if path is None:
                    missing.append(label)
            if missing:
                missing_report = "Missing label checkpoints: " + ", ".join(missing)
                continue

            _load_checkpoint_model(predictor["amp_checkpoint"], device)
            for label in FUNCTIONAL_LABELS:
                _load_checkpoint_model(predictor["label_checkpoints"][label], device)

            predictor["model_available"] = True
            return predictor

        predictor["warning"] = missing_report or "AMP checkpoint was not found."
        return predictor
    except Exception as exc:
        predictor["warning"] = str(exc)
        return predictor


def _build_prediction_feature_bundle(unique_sequences: pd.DataFrame, predictor: Dict) -> Tuple[pd.DataFrame, object]:
    build_unified_pkl = predictor["build_unified_pkl"]
    quad_dataset_cls = predictor["QuadOutputDataset"]
    collate_fn = predictor["quad_output_collate_fn"]
    torch_mod = predictor["torch"]

    tmp_df = unique_sequences[["sequence"]].copy()
    tmp_df["label"] = 0.0

    csv_handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    pkl_handle = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    csv_path = csv_handle.name
    pkl_path = pkl_handle.name
    csv_handle.close()
    pkl_handle.close()

    try:
        tmp_df.to_csv(csv_path, index=False)
        build_unified_pkl(
            csv_path=Path(csv_path),
            output_pkl=Path(pkl_path),
            sequence_col="sequence",
            label_cols=["label"],
            esm_model_name="esm2_t12_35M_UR50D",
            max_seq_len=700,
            device=predictor["device"],
            knowledge_transformer_ckpt=predictor["knowledge_transformer_ckpt"],
            knowledge_dim=256,
        )

        import pickle

        with open(pkl_path, "rb") as handle:
            raw_data = pickle.load(handle)

        rows = []
        for seq_hash, content in raw_data["features"].items():
            rows.append(
                {
                    "hash": seq_hash,
                    "sequence": content.get("sequence", ""),
                    "label": content.get("labels", [0])[0] if content.get("labels") else 0,
                }
            )
        feature_df = pd.DataFrame(rows)
        dataset = quad_dataset_cls(feature_df, feature_dict=raw_data["features"])
        loader = torch_mod.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        return feature_df, loader
    finally:
        for path in [csv_path, pkl_path]:
            try:
                os.remove(path)
            except Exception:
                pass


def _infer_probabilities(loader, checkpoint_path: str, predictor: Dict) -> List[float]:
    torch_mod = predictor["torch"]
    device = predictor["device"]
    loaded = _load_checkpoint_model(checkpoint_path, device)
    model = loaded["model"]

    probs = []
    with torch_mod.no_grad():
        for esm_feat, kn_feat, _ in loader:
            esm_feat = esm_feat.to(device)
            kn_feat = kn_feat.to(device)
            logits = model(esm_features=esm_feat, knowledge_features=kn_feat)
            batch_probs = torch_mod.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            probs.extend(batch_probs.tolist())
    return probs


def run_maple_prediction(valid_df: pd.DataFrame, predictor, thresholds: dict) -> pd.DataFrame:
    """Run MAPLE inference for valid sequences, or return NA columns if unavailable."""
    result_columns = ["sequence_id", "P_AMP", "AMP_label"]
    for label in FUNCTIONAL_LABELS:
        result_columns.extend([f"P_{label}", f"{label}_label"])

    if valid_df.empty:
        return pd.DataFrame(columns=result_columns)

    if not predictor or not predictor.get("model_available", False):
        out = valid_df[["sequence_id"]].copy()
        out["P_AMP"] = pd.NA
        out["AMP_label"] = pd.NA
        for label in FUNCTIONAL_LABELS:
            out[f"P_{label}"] = pd.NA
            out[f"{label}_label"] = pd.NA
        return out[result_columns]

    unique_sequences = valid_df[["sequence"]].drop_duplicates().reset_index(drop=True)
    feature_df, loader = _build_prediction_feature_bundle(unique_sequences, predictor)

    prediction_lookup = pd.DataFrame({"sequence": feature_df["sequence"].tolist()})

    amp_probs = _infer_probabilities(loader, predictor["amp_checkpoint"], predictor)
    threshold_amp = thresholds.get("amp", 0.5)
    prediction_lookup["P_AMP"] = amp_probs
    prediction_lookup["AMP_label"] = [int(prob >= threshold_amp) for prob in amp_probs]

    amp_positive_lookup = prediction_lookup[prediction_lookup["AMP_label"] == 1][["sequence"]].drop_duplicates().reset_index(drop=True)

    for label in FUNCTIONAL_LABELS:
        prediction_lookup[f"P_{label}"] = pd.NA
        prediction_lookup[f"{label}_label"] = pd.NA

    if not amp_positive_lookup.empty:
        amp_feature_df, amp_loader = _build_prediction_feature_bundle(amp_positive_lookup, predictor)
        functional_lookup = pd.DataFrame({"sequence": amp_feature_df["sequence"].tolist()})

        for label in FUNCTIONAL_LABELS:
            probs = _infer_probabilities(amp_loader, predictor["label_checkpoints"][label], predictor)
            threshold = thresholds.get(label, 0.5)
            functional_lookup[f"P_{label}"] = probs
            functional_lookup[f"{label}_label"] = [int(prob >= threshold) for prob in probs]

        prediction_lookup = prediction_lookup.merge(functional_lookup, on="sequence", how="left", suffixes=("", "_pred"))
        for label in FUNCTIONAL_LABELS:
            prob_col = f"P_{label}"
            label_col = f"{label}_label"
            pred_prob_col = f"{prob_col}_pred"
            pred_label_col = f"{label_col}_pred"
            if pred_prob_col in prediction_lookup.columns:
                prediction_lookup[prob_col] = prediction_lookup[pred_prob_col]
                prediction_lookup.drop(columns=[pred_prob_col], inplace=True)
            if pred_label_col in prediction_lookup.columns:
                prediction_lookup[label_col] = prediction_lookup[pred_label_col]
                prediction_lookup.drop(columns=[pred_label_col], inplace=True)

    merged = valid_df[["sequence_id", "sequence"]].merge(prediction_lookup, on="sequence", how="left")
    return merged[result_columns]


def assign_priority_group(row, thresholds: dict, model_available: bool) -> str:
    """Assign a prioritization group using validity and prediction constraints."""
    if not bool(row.get("valid", False)):
        return "invalid_sequence"
    if not model_available:
        return "properties_only"

    amp_prob = row.get("P_AMP")
    antibacterial_prob = row.get("P_antibacterial")
    hemolytic_prob = row.get("P_hemolytic")
    cytotoxic_prob = row.get("P_cytotoxic")
    anti_mammalian_prob = row.get("P_anti_mammalian_cells")

    if pd.isna(amp_prob):
        return "properties_only"

    threshold_amp = thresholds.get("amp", 0.5)
    threshold_antibacterial = thresholds.get("antibacterial", 0.5)
    threshold_hemolytic = thresholds.get("hemolytic", 0.5)
    threshold_cytotoxic = thresholds.get("cytotoxic", 0.5)
    threshold_anti_mammalian = thresholds.get("anti_mammalian_cells", 0.5)

    if (
        amp_prob >= threshold_amp
        and not pd.isna(antibacterial_prob)
        and antibacterial_prob >= threshold_antibacterial
        and not pd.isna(hemolytic_prob)
        and hemolytic_prob < threshold_hemolytic
    ):
        return "high_priority_selective"

    if (
        (not pd.isna(hemolytic_prob) and hemolytic_prob >= threshold_hemolytic)
        or (not pd.isna(cytotoxic_prob) and cytotoxic_prob >= threshold_cytotoxic)
        or (not pd.isna(anti_mammalian_prob) and anti_mammalian_prob >= threshold_anti_mammalian)
    ):
        return "effective_but_toxicity_flagged"

    if pd.isna(antibacterial_prob) or antibacterial_prob < threshold_antibacterial:
        return "low_antibacterial_potential"

    if amp_prob < threshold_amp:
        return "low_antibacterial_potential"

    return "intermediate_or_uncertain"


def dataframe_to_fasta(df: pd.DataFrame) -> str:
    """Convert a dataframe with sequence_id and sequence into FASTA text."""
    lines = []
    for _, row in df.iterrows():
        seq_id = str(row.get("sequence_id", "sequence")).strip() or "sequence"
        seq = str(row.get("sequence", "")).strip()
        if not seq:
            continue
        lines.append(f">{seq_id}")
        lines.append(seq)
    return "\n".join(lines)


def _render_summary(final_df: pd.DataFrame, model_available: bool) -> None:
    total = len(final_df)
    valid_count = int(final_df["valid"].fillna(False).sum()) if "valid" in final_df.columns else 0
    invalid_count = total - valid_count
    predicted_amp = int((final_df.get("AMP_label").fillna(0) == 1).sum()) if "AMP_label" in final_df.columns and model_available else 0
    high_priority = int((final_df.get("priority_group") == "high_priority_selective").sum()) if "priority_group" in final_df.columns else 0
    possible_toxic = int((final_df.get("priority_group") == "effective_but_toxicity_flagged").sum()) if "priority_group" in final_df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sequences submitted", total)
    c2.metric("Usable sequences", valid_count)
    c3.metric("Skipped sequences", invalid_count)
    c4.metric("Predicted AMP candidates", predicted_amp)
    c5.metric("High-priority candidates", high_priority)
    st.caption(f"Possible toxicity concerns: {possible_toxic}")


def _render_visualizations(final_df: pd.DataFrame, model_available: bool) -> None:
    if final_df.empty:
        return

    efficacy_col = _display_name_for_column("P_antibacterial")
    hemolysis_col = _display_name_for_column("P_hemolytic")
    hydrophobicity_col = _display_name_for_column("mean_hydrophobicity_kyte_doolittle")
    charge_col = _display_name_for_column("net_charge_approx")

    st.subheader("Charts")
    chart_df = final_df.copy()
    if "priority_group" in chart_df.columns:
        chart_df["priority_group_display"] = chart_df["priority_group"].fillna("unknown").map(_pretty_priority_group)
        chart_df["priority_group_chart"] = chart_df["priority_group_display"].map(_chart_priority_group)

    scatter_df = chart_df.dropna(subset=["net_charge_approx", "mean_hydrophobicity_kyte_doolittle"]).copy()
    if model_available and {"P_antibacterial", "P_hemolytic"}.issubset(chart_df.columns):
        pred_scatter_df = chart_df.dropna(subset=["P_antibacterial", "P_hemolytic"]).copy()
        if not pred_scatter_df.empty:
            pred_scatter_df = pred_scatter_df.rename(
                columns={
                    "P_antibacterial": efficacy_col,
                    "P_hemolytic": hemolysis_col,
                }
            )
            threshold_antibacterial = 0.5
            threshold_hemolytic = 0.5
            if "thresholds_runtime" in st.session_state:
                threshold_antibacterial = st.session_state["thresholds_runtime"].get("antibacterial", 0.5)
                threshold_hemolytic = st.session_state["thresholds_runtime"].get("hemolytic", 0.5)
            if px is not None:
                fig = px.scatter(
                    pred_scatter_df,
                    x=efficacy_col,
                    y=hemolysis_col,
                    color="priority_group_display" if "priority_group_display" in pred_scatter_df.columns else None,
                    hover_name="sequence_id",
                    title="Efficacy–toxicity prioritization map",
                    labels={
                        efficacy_col: efficacy_col,
                        hemolysis_col: hemolysis_col,
                        "priority_group_display": "Recommendation group",
                    },
                )
                fig.add_vline(x=threshold_antibacterial, line_dash="dash", line_color="green")
                fig.add_hline(y=threshold_hemolytic, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Efficacy–toxicity prioritization map")
                st.scatter_chart(pred_scatter_df[[efficacy_col, hemolysis_col]])

    if not scatter_df.empty:
        scatter_df = scatter_df.rename(
            columns={
                "mean_hydrophobicity_kyte_doolittle": hydrophobicity_col,
                "net_charge_approx": charge_col,
            }
        )
        if px is not None:
            fig = px.scatter(
                scatter_df,
                x=hydrophobicity_col,
                y=charge_col,
                color="priority_group_display" if "priority_group_display" in scatter_df.columns else None,
                hover_name="sequence_id",
                title="Physicochemical selectivity map",
                labels={
                    charge_col: charge_col,
                    hydrophobicity_col: hydrophobicity_col,
                    "priority_group_display": "Recommendation group",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Physicochemical selectivity map")
            st.scatter_chart(scatter_df[[hydrophobicity_col, charge_col]])

    if model_available and "priority_group" in chart_df.columns:
        counts = chart_df["priority_group_display"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["priority_group_display", "count"]
        counts["priority_group_chart"] = counts["priority_group_display"].map(_chart_priority_group)
        if px is not None:
            fig = px.bar(
                counts,
                x="count",
                y="priority_group_chart",
                orientation="h",
                title="Recommendation group distribution",
                labels={"priority_group_chart": "Recommendation group", "count": "Number of sequences"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fallback_counts = counts.set_index("priority_group_chart")["count"].rename("Number of sequences")
            st.bar_chart(fallback_counts)
    else:
        length_df = final_df.dropna(subset=["length"]).copy()
        if not length_df.empty:
            if px is not None:
                fig = px.histogram(
                    length_df,
                    x="length",
                    nbins=30,
                    title="Sequence length distribution",
                    labels={"length": "Length (aa)", "count": "Number of sequences"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                length_counts = length_df["length"].value_counts().sort_index().rename("count").to_frame()
                st.bar_chart(length_counts)


def _prepare_final_output(input_df: pd.DataFrame, prediction_df: pd.DataFrame, thresholds: Dict[str, float], model_available: bool) -> pd.DataFrame:
    base = input_df.copy()
    base["clean_sequence"] = base["sequence"].map(clean_sequence)
    validation = base["clean_sequence"].map(validate_sequence)
    base["valid"] = validation.map(lambda x: x[0])
    base["invalid_reason"] = validation.map(lambda x: x[1] or "")
    base["length_warning"] = base["clean_sequence"].map(_length_warning)

    property_df = base["clean_sequence"].map(compute_sequence_properties).apply(pd.Series)
    base = pd.concat([base, property_df], axis=1)
    raw56_df = base["clean_sequence"].map(compute_merged_raw56_properties).apply(pd.Series)
    base = pd.concat([base, raw56_df], axis=1)

    merged = base.merge(prediction_df, on=["sequence_id"], how="left")
    if "P_AMP" not in merged.columns:
        merged["P_AMP"] = pd.NA
        merged["AMP_label"] = pd.NA
    for label in FUNCTIONAL_LABELS:
        if f"P_{label}" not in merged.columns:
            merged[f"P_{label}"] = pd.NA
        if f"{label}_label" not in merged.columns:
            merged[f"{label}_label"] = pd.NA

    merged["selectivity_score"] = merged.apply(
        lambda row: (
            float(row["P_antibacterial"]) - float(row["P_hemolytic"])
            if not pd.isna(row["P_antibacterial"]) and not pd.isna(row["P_hemolytic"])
            else None
        ),
        axis=1,
    )
    merged["priority_group"] = merged.apply(
        lambda row: assign_priority_group(row, thresholds=thresholds, model_available=model_available),
        axis=1,
    )

    motif_df, _ = _load_motif_reference()
    motif_features = merged["clean_sequence"].map(lambda seq: _scan_sequence_motifs(seq, motif_df)).apply(pd.Series)
    merged = pd.concat([merged, motif_features], axis=1)
    merged["interpretation"] = merged.apply(lambda row: _generate_interpretation(row, model_available=model_available), axis=1)

    ordered_columns = [
        "sequence_id",
        "sequence",
        "clean_sequence",
        "valid",
        "invalid_reason",
        "length_warning",
        "P_AMP",
        "AMP_label",
        "P_antibacterial",
        "antibacterial_label",
        "P_hemolytic",
        "hemolytic_label",
        "P_cytotoxic",
        "cytotoxic_label",
        "P_anti_mammalian_cells",
        "anti_mammalian_cells_label",
        "selectivity_score",
        "priority_group",
        "interpretation",
        "n_antibacterial_selective_motifs",
        "n_hemolytic_associated_motifs",
        "n_dual_activity_motifs",
        "motif_balance_score",
        "top_matched_motifs",
    ]
    for label in FUNCTIONAL_LABELS:
        col_prob = f"P_{label}"
        col_label = f"{label}_label"
        if col_prob not in ordered_columns:
            ordered_columns.extend([col_prob, col_label])

    ordered_columns.extend(
        [
            "length",
            "molecular_weight_approx",
            "net_charge_approx",
            "charge_density",
            "mean_hydrophobicity_kyte_doolittle",
            "fraction_positive",
            "fraction_negative",
            "fraction_hydrophobic",
            "fraction_polar",
            "fraction_aromatic",
            "fraction_small",
            "fraction_glycine",
            "fraction_proline",
        ]
    )
    ordered_columns.extend(RAW56_UNIQUE_COLUMNS)

    remaining = [col for col in merged.columns if col not in ordered_columns]
    return merged[ordered_columns + remaining]


def main():
    """Run the Streamlit MAPLE demo."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)
    st.warning("Predictions are computational estimates and require experimental validation.")
    st.info(APP_SELECTIVITY_NOTE)
    st.caption(
        "Functional activity prediction is performed only for sequences classified as AMP-positive by the AMP screening model. "
        "Sequences that do not pass the AMP screen are reported with AMP probability and sequence-derived descriptors only."
    )
    st.caption(
        "Physicochemical descriptors are approximate sequence-derived descriptors for interpretation only and are not experimental measurements."
    )

    with st.sidebar:
        st.header("Settings")
        input_mode = st.radio("How would you like to provide sequences?", ["Manual", "CSV", "FASTA"], index=0)
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        default_checkpoint_dir = _default_checkpoint_dir()
        default_threshold_file = _default_threshold_file()
        checkpoint_dir = st.text_input("Model folder", value=default_checkpoint_dir)
        threshold_file = st.text_input("Threshold file", value=default_threshold_file)
        st.caption(
            "Model files should be placed under the folder specified here. "
            "Expected layout: AMP.pt; label checkpoints at label/<label>.pt, label/<label>/<label>.pt, or <label>.pt; "
            "knowledge_transformer.pt; thresholds.json. "
            "The default folder can be changed if your checkpoints are stored elsewhere."
        )
        show_debug = st.checkbox("Show debug details", value=False)
        run_clicked = st.button("Run prediction", use_container_width=True)

    resolved_device = resolve_device(device_choice)
    if device_choice == "cuda" and resolved_device == "cpu":
        st.warning("CUDA was requested but is not available. Falling back to CPU.")

    input_df = pd.DataFrame(columns=["sequence_id", "sequence"])
    csv_sequence_col = None
    csv_raw_df = None
    parse_error = None

    st.subheader("Sequence input")
    if input_mode == "Manual":
        manual_text = st.text_area(
            "Enter one peptide sequence per line, or paste FASTA text.",
            height=220,
            placeholder="KWKLFKKIGAVLKVL\n>peptide_2\nGIGKFLHSAKKFGKAFVGEIMNS",
        )
        if run_clicked:
            try:
                input_df = parse_manual_input(manual_text)
            except Exception as exc:
                parse_error = exc

    elif input_mode == "CSV":
        uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_csv is not None:
            try:
                csv_raw_df = _parse_csv_bytes(uploaded_csv.getvalue())
                if csv_raw_df.empty:
                    st.warning("The uploaded CSV file is empty.")
                else:
                    columns = list(csv_raw_df.columns)
                    lower_map = {str(col).lower(): col for col in columns}
                    default_col = None
                    for candidate in LIKELY_SEQUENCE_COLUMNS:
                        if candidate in lower_map:
                            default_col = lower_map[candidate]
                            break
                    default_index = columns.index(default_col) if default_col in columns else 0
                    csv_sequence_col = st.selectbox("Which column contains the peptide sequences?", columns, index=default_index)
                    st.caption(f"Rows detected in the CSV file: {len(csv_raw_df)}")
            except Exception as exc:
                parse_error = exc
        elif run_clicked:
            st.warning("Please upload a CSV file before running prediction.")

        if run_clicked and parse_error is None and uploaded_csv is not None:
            try:
                if csv_raw_df is None or csv_raw_df.empty:
                    input_df = pd.DataFrame(columns=["sequence_id", "sequence"])
                elif csv_sequence_col:
                    input_df = _coerce_csv_dataframe(csv_raw_df, sequence_col=csv_sequence_col)
                else:
                    input_df = parse_csv_upload(uploaded_csv)
            except Exception as exc:
                parse_error = exc

    else:
        uploaded_fasta = st.file_uploader("Upload a FASTA file", type=["fa", "fasta", "faa", "txt"])
        if run_clicked and uploaded_fasta is not None:
            try:
                fasta_text = uploaded_fasta.getvalue().decode("utf-8", errors="ignore")
                input_df = parse_fasta_text(fasta_text)
            except Exception as exc:
                parse_error = exc
        elif run_clicked:
            st.warning("Please upload a FASTA file before running prediction.")

    if parse_error is not None:
        st.error(f"Input parsing failed: {parse_error}")
        if show_debug:
            st.exception(parse_error)
        return

    if not run_clicked:
        st.info("Configure the input mode, then click Run prediction.")
        return

    if input_df.empty:
        st.warning("No sequences were parsed from the provided input.")
        return

    if len(input_df) > 200:
        st.warning("Large batches may be slow in the Streamlit interface. Command-line prediction is recommended for batch analysis.")

    thresholds, threshold_source = load_thresholds(threshold_file)
    st.session_state["thresholds_runtime"] = thresholds
    st.caption(f"Threshold source: {_pretty_threshold_source(threshold_source)}")
    motif_df, motif_message = _load_motif_reference()
    if motif_df is None:
        st.caption(motif_message)
    else:
        st.caption("Motif-based interpretation is enabled.")

    predictor = try_load_maple_predictor(checkpoint_dir, resolved_device)
    model_available = bool(predictor.get("model_available", False))
    if not model_available:
        st.warning(
            "MAPLE model checkpoints were not found or could not be loaded. "
            "Only sequence-derived descriptors are shown."
        )
        if predictor.get("warning"):
            st.caption(f"Model loading note: {predictor['warning']}")

    cleaned_input = input_df.copy()
    cleaned_input["sequence"] = cleaned_input["sequence"].astype(str).str.strip()
    cleaned_input["sequence_id"] = cleaned_input["sequence_id"].astype(str).str.strip()

    valid_mask = cleaned_input["sequence"].map(clean_sequence).map(validate_sequence).map(lambda item: item[0])
    valid_df = cleaned_input.loc[valid_mask, ["sequence_id", "sequence"]].copy()
    valid_df["sequence"] = valid_df["sequence"].map(clean_sequence)

    try:
        prediction_df = run_maple_prediction(valid_df, predictor, thresholds)
    except Exception as exc:
        prediction_df = run_maple_prediction(valid_df, None, thresholds)
        model_available = False
        st.warning(
            "MAPLE model checkpoints were not found or could not be loaded. "
            "Only sequence-derived descriptors are shown."
        )
        st.caption(f"Prediction note: {exc}")
        if show_debug:
            st.exception(exc)

    final_df = _prepare_final_output(cleaned_input, prediction_df, thresholds, model_available=model_available)
    display_df = _build_display_dataframe(final_df)

    st.subheader("Summary")
    _render_summary(final_df, model_available=model_available)
    st.info(_build_summary_text(final_df, model_available=model_available))

    st.subheader("Detailed results")
    compact_columns = _select_display_columns(display_df)
    show_full_table = st.checkbox("Show the full table with all measured and predicted fields", value=False)
    if show_full_table:
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(display_df[compact_columns], use_container_width=True)
    raw56_columns = _select_raw56_display_columns(display_df)
    if raw56_columns:
        with st.expander("Additional sequence descriptors"):
            st.dataframe(display_df[raw56_columns], use_container_width=True)

    _render_visualizations(final_df, model_available=model_available)

    st.subheader("Download")
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download the results table as CSV",
        data=csv_bytes,
        file_name="maple_results.csv",
        mime="text/csv",
    )

    valid_fasta_df = final_df[final_df["valid"] == True][["sequence_id", "clean_sequence"]].copy()
    valid_fasta_df = valid_fasta_df.rename(columns={"clean_sequence": "sequence"})
    fasta_text = dataframe_to_fasta(valid_fasta_df)
    st.download_button(
        "Download usable sequences as FASTA",
        data=fasta_text.encode("utf-8"),
        file_name="valid_sequences.fasta",
        mime="text/plain",
    )

    with st.expander("Advanced information"):
        st.write(f"Computation device in use: `{resolved_device}`")
        st.write("Model files are loaded from the selected model folder.")
        st.write("Decision settings are loaded from the file specified in the sidebar.")
        st.write(
            "Expected checkpoint layout: `AMP.pt`; label checkpoints at `label/<label>.pt`, "
            "`label/<label>/<label>.pt`, or `<label>.pt`; `knowledge_transformer.pt`; `thresholds.json`."
        )
        st.write("This page reuses the existing MAPLE prediction code in the project whenever available.")


if __name__ == "__main__":
    main()
