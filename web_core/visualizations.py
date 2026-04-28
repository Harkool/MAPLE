import pandas as pd
import streamlit as st

from web_core.ui_text import (
    _chart_priority_group,
    _display_name_for_column,
    _pretty_priority_group,
)

try:
    import plotly.express as px
except Exception:
    px = None


def render_summary(final_df: pd.DataFrame, model_available: bool) -> None:
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


def render_visualizations(final_df: pd.DataFrame, model_available: bool) -> None:
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
