import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "articles.db"
CSV_PATH = DATA_DIR / "articles.csv"

st.set_page_config(page_title="Abstract Quality & Citations", page_icon="📚", layout="wide")
st.title("📚 NLP-Driven Evaluation of Abstract Quality")
st.caption("Explore abstract structure, readability proxies, and citation-count associations in Q1 journal records.")

@st.cache_data
def load_data():
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as con:
            return pd.read_sql_query("SELECT * FROM articles", con)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame()

def score_abstracts(frame):
    out = frame.copy()
    text = out["abstract"].fillna("").astype(str)
    out["word_count"] = text.str.findall(r"\b[\w'-]+\b").str.len()
    out["sentence_count"] = text.str.count(r"[.!?]+").clip(lower=1)
    out["avg_words_per_sentence"] = out["word_count"] / out["sentence_count"]
    out["has_background"] = text.str.contains(r"\bbackground\b", case=False, regex=True).astype(int)
    out["has_objective"] = text.str.contains(r"\b(objective|aim|purpose)\b", case=False, regex=True).astype(int)
    out["has_methods"] = text.str.contains(r"\b(methods?|methodology|approach|we compare|we train|we analyze|we searched)\b", case=False, regex=True).astype(int)
    out["has_results"] = text.str.contains(r"\b(results?|findings?|improves?|achieves?)\b", case=False, regex=True).astype(int)
    out["has_conclusion"] = text.str.contains(r"\b(conclusion|conclude|suggest|indicate|these findings)\b", case=False, regex=True).astype(int)
    component_cols = ["has_background", "has_objective", "has_methods", "has_results", "has_conclusion"]
    out["structure_score"] = out[component_cols].sum(axis=1) / len(component_cols) * 100
    # A transparent heuristic score, not a validated scientific measure.
    length_score = (100 - (out["word_count"] - 200).abs() / 2).clip(0, 100)
    sentence_score = (100 - (out["avg_words_per_sentence"] - 20).abs() * 2).clip(0, 100)
    out["heuristic_quality_score"] = (0.7 * out["structure_score"] + 0.2 * length_score + 0.1 * sentence_score).round(1)
    out["log_citations"] = np.log1p(pd.to_numeric(out["citations"], errors="coerce").fillna(0))
    return out

def correlation(x, y, fn):
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan, np.nan
    return fn(valid["x"], valid["y"])

df = st.session_state.get("uploaded_df", load_data())
if df.empty:
    st.warning("No dataset found. Upload a CSV with title, abstract, year, journal, quartile, subject_area, citations columns.")
    upload = st.file_uploader("Upload article CSV", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
    else:
        st.stop()

required = {"title", "abstract", "citations"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

for col, default in [("year", 0), ("journal", "Unknown"), ("quartile", "Q1"), ("subject_area", "Unknown"), ("source", "uploaded")]:
    if col not in df.columns:
        df[col] = default
df["citations"] = pd.to_numeric(df["citations"], errors="coerce")
df = df.dropna(subset=["citations"]).copy()
scored = score_abstracts(df)

with st.sidebar:
    st.header("Filters")
    years = sorted(pd.to_numeric(scored["year"], errors="coerce").dropna().astype(int).unique().tolist())
    selected_years = st.multiselect("Publication year", years, default=years)
    subjects = sorted(scored["subject_area"].fillna("Unknown").astype(str).unique())
    selected_subjects = st.multiselect("Subject area", subjects, default=subjects)
    quartiles = sorted(scored["quartile"].fillna("Unknown").astype(str).unique())
    selected_quartiles = st.multiselect("Journal quartile", quartiles, default=quartiles)
    min_citations = st.number_input("Minimum citations", min_value=0, value=0, step=1)

filtered = scored[
    pd.to_numeric(scored["year"], errors="coerce").fillna(0).astype(int).isin(selected_years)
    & scored["subject_area"].fillna("Unknown").astype(str).isin(selected_subjects)
    & scored["quartile"].fillna("Unknown").astype(str).isin(selected_quartiles)
    & (scored["citations"] >= min_citations)
].copy()

if filtered.empty:
    st.info("No records match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles", f"{len(filtered):,}")
c2.metric("Mean citations", f"{filtered['citations'].mean():.1f}")
c3.metric("Mean quality score", f"{filtered['heuristic_quality_score'].mean():.1f}/100")
c4.metric("Median abstract words", f"{filtered['word_count'].median():.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlation", "Explore articles", "Upload & export"])
with tab1:
    left, right = st.columns(2)
    with left:
        fig = px.scatter(filtered, x="heuristic_quality_score", y="citations", color="subject_area",
                         hover_data=["title", "year", "journal", "word_count"],
                         title="Heuristic abstract quality vs citations")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(filtered, x="word_count", nbins=20, title="Abstract word-count distribution")
        st.plotly_chart(fig, use_container_width=True)
    by_year = filtered.groupby("year", dropna=False).agg(articles=("title", "count"), mean_citations=("citations", "mean")).reset_index()
    st.plotly_chart(px.line(by_year, x="year", y="mean_citations", markers=True, title="Mean citations by publication year"), use_container_width=True)
    st.caption("Citation counts are age- and field-dependent; use year/subject filters and interpret associations cautiously.")
with tab2:
    xcol = st.selectbox("Abstract metric", ["heuristic_quality_score", "structure_score", "word_count", "avg_words_per_sentence"])
    pearson, p_pearson = correlation(filtered[xcol], filtered["log_citations"], pearsonr)
    spearman, p_spearman = correlation(filtered[xcol], filtered["citations"], spearmanr)
    a, b, c, d = st.columns(4)
    a.metric("Pearson r (log citations)", "—" if np.isnan(pearson) else f"{pearson:.3f}")
    b.metric("Pearson p-value", "—" if np.isnan(p_pearson) else f"{p_pearson:.4f}")
    c.metric("Spearman ρ (citations)", "—" if np.isnan(spearman) else f"{spearman:.3f}")
    d.metric("Spearman p-value", "—" if np.isnan(p_spearman) else f"{p_spearman:.4f}")
    fig = px.scatter(filtered, x=xcol, y="log_citations", trendline="ols",
                     hover_data=["title", "year", "journal"], title=f"{xcol} vs log(1 + citations)")
    st.plotly_chart(fig, use_container_width=True)
    st.info("These are unadjusted correlations, not causal effects. For publishable analysis, control for publication age, field, journal, and document type; use real, verified bibliographic data.")
with tab3:
    search = st.text_input("Search title, abstract, or journal")
    view = filtered.copy()
    if search:
        mask = (
            view["title"].fillna("").str.contains(search, case=False, regex=False)
            | view["abstract"].fillna("").str.contains(search, case=False, regex=False)
            | view["journal"].fillna("").str.contains(search, case=False, regex=False)
        )
        view = view[mask]
    show_cols = [c for c in ["article_id", "title", "year", "journal", "quartile", "subject_area", "citations", "word_count", "structure_score", "heuristic_quality_score", "source"] if c in view]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
    with st.expander("Read selected abstracts"):
        if not view.empty:
            selected = st.selectbox("Article", view["title"].tolist())
            row = view[view["title"] == selected].iloc[0]
            st.write(row["abstract"])
with tab4:
    uploaded = st.file_uploader("Add another CSV dataset", type=["csv"], key="extra_csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(new_df)} rows. Validate required columns before combining.")
        if required.issubset(new_df.columns):
            if st.button("Analyze uploaded dataset instead"):
                st.session_state["uploaded_df"] = new_df
                st.success("Uploaded dataset is now active for this session.")
    st.download_button("Download filtered scored data (CSV)", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="abstract_quality_scored.csv", mime="text/csv")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="Scored articles")
    st.download_button("Download filtered data (Excel)", data=excel_buffer.getvalue(),
                       file_name="abstract_quality_scored.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Dataset provenance")
    st.write(filtered["source"].value_counts(dropna=False).rename_axis("source").reset_index(name="records"))

st.sidebar.markdown("---")
st.sidebar.caption("Demo dataset is synthetic. Replace with verified Scopus-exported data before drawing research conclusions.")
