from typing import List


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

AA_ORDER_56 = "ACDEFGHIKLMNPQRSTVWY"
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


def _pretty_yes_no_na(value) -> str:
    import pandas as pd

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


def raw56_display_columns() -> List[str]:
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
    return preferred
