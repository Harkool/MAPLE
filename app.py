from pathlib import Path

import pandas as pd
import streamlit as st

from web_core.input_parsing import (
    _parse_csv_bytes,
    coerce_csv_dataframe,
    parse_csv_upload,
    parse_fasta_text,
    parse_manual_input,
)
from web_core.predictor_runtime import (
    load_thresholds,
    resolve_device,
    run_maple_prediction,
    try_load_maple_predictor,
)
from web_core.results_table import (
    build_display_dataframe,
    build_summary_text,
    dataframe_to_fasta,
    load_motif_reference,
    prepare_final_output,
    select_additional_descriptor_columns,
    select_display_columns,
)
from web_core.ui_text import (
    APP_DESCRIPTION,
    APP_SELECTIVITY_NOTE,
    APP_TITLE,
    LIKELY_ID_COLUMNS,
    LIKELY_SEQUENCE_COLUMNS,
    _pretty_threshold_source,
)
from web_core.sequence_features import clean_sequence, validate_sequence
from web_core.visualizations import render_summary, render_visualizations

try:
    import torch
except Exception:
    torch = None

APP_DIR = Path(__file__).resolve().parent


def _default_checkpoint_dir() -> str:
    return "MAPLE_checkpoints"


def _default_threshold_file() -> str:
    checkpoint_dir = _default_checkpoint_dir()
    return f"{checkpoint_dir}/thresholds.json"




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

    resolved_device = resolve_device(device_choice, torch)
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
                    input_df = coerce_csv_dataframe(
                        csv_raw_df,
                        sequence_col=csv_sequence_col,
                        likely_sequence_columns=LIKELY_SEQUENCE_COLUMNS,
                        likely_id_columns=LIKELY_ID_COLUMNS,
                    )
                else:
                    input_df = parse_csv_upload(
                        uploaded_csv,
                        likely_sequence_columns=LIKELY_SEQUENCE_COLUMNS,
                        likely_id_columns=LIKELY_ID_COLUMNS,
                    )
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

    thresholds, threshold_source = load_thresholds(APP_DIR, threshold_file)
    st.session_state["thresholds_runtime"] = thresholds
    st.caption(f"Threshold source: {_pretty_threshold_source(threshold_source)}")
    motif_df, motif_message = load_motif_reference(APP_DIR)
    if motif_df is None:
        st.caption(motif_message)
    else:
        st.caption("Motif-based interpretation is enabled.")

    predictor = try_load_maple_predictor(APP_DIR, checkpoint_dir, resolved_device)
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

    final_df = prepare_final_output(
        cleaned_input,
        prediction_df,
        thresholds,
        model_available=model_available,
        motif_df=motif_df,
    )
    display_df = build_display_dataframe(final_df)

    st.subheader("Summary")
    render_summary(final_df, model_available=model_available)
    st.info(build_summary_text(final_df, model_available=model_available))

    st.subheader("Detailed results")
    compact_columns = select_display_columns(display_df)
    show_full_table = st.checkbox("Show the full table with all measured and predicted fields", value=False)
    if show_full_table:
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(display_df[compact_columns], use_container_width=True)
    raw56_columns = select_additional_descriptor_columns(display_df)
    if raw56_columns:
        with st.expander("Additional sequence descriptors"):
            st.dataframe(display_df[raw56_columns], use_container_width=True)

    render_visualizations(final_df, model_available=model_available)

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
