from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from web_core.sequence_features import (
    clean_sequence,
    compute_merged_raw56_properties,
    compute_sequence_properties,
    length_warning,
    validate_sequence,
)
from web_core.ui_text import (
    AA_ORDER_56,
    FUNCTIONAL_LABELS,
    RAW56_UNIQUE_COLUMNS,
    _pretty_binary_name,
    _pretty_invalid_reason,
    _pretty_label_name,
    _pretty_length_warning,
    _pretty_priority_group,
    _pretty_probability_name,
    _pretty_yes_no_na,
    raw56_display_columns,
)


@st.cache_data(show_spinner=False)
def load_motif_reference(app_dir: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Load optional motif interpretation reference if available."""
    motif_path = app_dir / "motif_reference.csv"
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


def scan_sequence_motifs(sequence: str, motif_df: Optional[pd.DataFrame]) -> Dict[str, object]:
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


def generate_interpretation(row, model_available: bool) -> str:
    """Generate a short candidate interpretation."""
    if not bool(row.get("valid", False)):
        reason = row.get("invalid_reason", "")
        if not reason:
            return "Invalid sequence: prediction was skipped."
        return f"Invalid sequence: {_pretty_invalid_reason(reason)}."

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


def build_display_dataframe(final_df: pd.DataFrame) -> pd.DataFrame:
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


def build_summary_text(final_df: pd.DataFrame, model_available: bool) -> str:
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


def select_display_columns(display_df: pd.DataFrame) -> List[str]:
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


def select_additional_descriptor_columns(display_df: pd.DataFrame) -> List[str]:
    return [col for col in raw56_display_columns() if col in display_df.columns]


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


def prepare_final_output(
    input_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    thresholds: Dict[str, float],
    model_available: bool,
    motif_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    base = input_df.copy()
    base["clean_sequence"] = base["sequence"].map(clean_sequence)
    validation = base["clean_sequence"].map(validate_sequence)
    base["valid"] = validation.map(lambda x: x[0])
    base["invalid_reason"] = validation.map(lambda x: x[1] or "")
    base["length_warning"] = base["clean_sequence"].map(length_warning)

    property_df = base["clean_sequence"].map(compute_sequence_properties).apply(pd.Series)
    base = pd.concat([base, property_df], axis=1)
    descriptor_df = base["clean_sequence"].map(compute_merged_raw56_properties).apply(pd.Series)
    base = pd.concat([base, descriptor_df], axis=1)

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

    motif_features = merged["clean_sequence"].map(lambda seq: scan_sequence_motifs(seq, motif_df)).apply(pd.Series)
    merged = pd.concat([merged, motif_features], axis=1)
    merged["interpretation"] = merged.apply(
        lambda row: generate_interpretation(row, model_available=model_available),
        axis=1,
    )

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
