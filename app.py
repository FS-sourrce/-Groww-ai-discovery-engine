import ast
import re
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Groww AI Discovery Engine", page_icon="🔎", layout="wide")
DATA_PATH = Path(__file__).parent / "research_dataset.csv"
SIP_RESEARCH_PATH = Path(__file__).parent / "groww_sip_research_starter.csv"

THEMES = {
    "App reliability": ["crash", "crashes", "crashing", "glitch", "glitches", "stuck", "hang", "slow", "loading", "not opening", "not working", "bug", "error", "freeze"],
    "Usability & navigation": ["confusing", "difficult to use", "easy to use", "interface", "navigation", "user friendly", "layout", "find", "understand", "complicated"],
    "Trust & support": ["customer support", "support", "complaint", "fraud", "scam", "trust", "response", "resolved", "help", "service"],
    "Fees & transactions": ["charges", "fee", "fees", "withdraw", "withdrawal", "deposit", "payment", "transaction", "refund", "bank", "mandate", "autopay", "deducted"],
    "Investment understanding": ["sip", "mutual fund", "investment", "investing", "portfolio", "returns", "return", "loss", "profit", "market", "risk", "nav", "fund"],
    "Performance & information": ["chart", "data", "price", "performance", "statement", "history", "update", "incorrect", "wrong", "display", "information"],
    "Positive experience": ["good", "great", "excellent", "best", "awesome", "love", "helpful", "smooth", "easy", "nice", "amazing"]
}
SIP_TERMS = ["sip", "systematic investment", "instalment", "installment", "mandate", "autopay", "pause sip", "stop sip", "cancel sip", "sip stopped", "sip failed"]
STOP_TERMS = ["stop sip", "stopped sip", "cancel sip", "pause sip", "paused sip", "discontinue", "stopped investing", "stop investing", "withdraw all", "redeem all"]
CONTINUE_TERMS = ["continue sip", "continued sip", "keep investing", "long term", "long-term", "monthly investment", "regular investment"]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).fillna("")
    for col in ["source", "record_type", "title", "text", "url", "date", "author", "rating"]:
        if col not in df.columns:
            df[col] = ""
    df["text"] = df["text"].astype(str)
    df["search_blob"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.lower()
    df["theme"] = df.apply(classify_theme, axis=1)
    df["sip_evidence"] = df["text"].apply(classify_sip)
    return df

def classify_theme(row):
    txt = (str(row.get("title", "")) + " " + str(row.get("text", ""))).lower()
    scores = {theme: sum(1 for term in terms if term in txt) for theme, terms in THEMES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Needs review"

def classify_sip(text):
    txt = str(text).lower()
    if any(t in txt for t in STOP_TERMS):
        return "Possible pause/stop signal"
    if any(t in txt for t in CONTINUE_TERMS):
        return "Possible continuation signal"
    if any(t in txt for t in SIP_TERMS):
        return "SIP mentioned, behaviour unclear"
    return "No direct SIP evidence detected"

def has_term(text, term):
    return term.lower().strip() in str(text).lower() if term.strip() else True

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load research_dataset.csv: {e}")
    st.stop()

st.title("Groww AI Discovery Engine")
st.caption("Public feedback explorer | Early MVP | Exploratory classifications require human validation")
st.info("Evidence boundary: app reviews can reveal product friction, but they do not prove why an investor continued or stopped a SIP. YouTube video leads are not comments. This prototype does not calculate actual 12-month SIP retention.")

with st.sidebar:
    st.header("Explore feedback")
    query = st.text_input("Search words or a research question", placeholder="e.g. SIP mandate failed")
    sources = sorted([x for x in df["source"].astype(str).unique() if x.strip()])
    chosen_sources = st.multiselect("Source", sources, default=sources)
    themes = sorted(df["theme"].unique().tolist())
    chosen_themes = st.multiselect("Candidate theme", themes, default=themes)
    sip_options = sorted(df["sip_evidence"].unique().tolist())
    chosen_sip = st.multiselect("SIP evidence label", sip_options, default=sip_options)
    types = sorted([x for x in df["record_type"].astype(str).unique() if x.strip()])
    chosen_types = st.multiselect("Record type", types, default=types)
    min_rating, max_rating = st.slider("Rating range (where available)", 1, 5, (1, 5))

filtered = df[df["source"].isin(chosen_sources) & df["theme"].isin(chosen_themes) & df["sip_evidence"].isin(chosen_sip) & df["record_type"].isin(chosen_types)].copy()
if query.strip():
    terms = [t for t in re.findall(r"[\w'-]+", query.lower()) if len(t) > 2 and t not in {"why", "what", "when", "where", "does", "after", "with", "from", "that", "they", "them", "their", "into", "about", "have", "been", "are", "the", "and", "for"}]
    if terms:
        filtered["search_score"] = filtered["search_blob"].apply(lambda s: sum(s.count(t) for t in terms))
        filtered = filtered[filtered["search_score"] > 0].sort_values("search_score", ascending=False)
    else:
        filtered["search_score"] = 0

# Top-level metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in dataset", f"{len(df):,}")
c2.metric("Matching records", f"{len(filtered):,}")
c3.metric("Distinct sources", str(df["source"].nunique()))
c4.metric("Records mentioning SIP", str(df["sip_evidence"].ne("No direct SIP evidence detected").sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Feedback Explorer", "Theme Overview", "Evidence & Validation", "SIP Research"])
with tab1:
    st.subheader("Search results")
    if filtered.empty:
        st.warning("No matching records. Try a broader search or remove a filter.")
    else:
        show_cols = ["source", "record_type", "date", "rating", "theme", "sip_evidence", "title", "text", "url"]
        st.dataframe(filtered[show_cols].rename(columns={"source":"Source", "record_type":"Record type", "date":"Date", "rating":"Rating", "theme":"Candidate theme", "sip_evidence":"SIP evidence label", "title":"Title", "text":"Original feedback", "url":"Source URL"}), use_container_width=True, hide_index=True, column_config={"Source URL": st.column_config.LinkColumn("Source URL", display_text="Open source"), "Original feedback": st.column_config.TextColumn(width="large")})
        st.download_button("Download filtered evidence (CSV)", filtered[show_cols].to_csv(index=False).encode("utf-8"), "groww_filtered_evidence.csv", "text/csv")
        st.caption("Candidate themes and SIP labels are heuristic suggestions, not validated AI findings.")
with tab2:
    st.subheader("What appears in this dataset")
    theme_counts = filtered["theme"].value_counts().rename_axis("Candidate theme").reset_index(name="Records")
    if not theme_counts.empty:
        st.bar_chart(theme_counts.set_index("Candidate theme"))
        st.dataframe(theme_counts, use_container_width=True, hide_index=True)
    st.markdown("#### SIP evidence labels")
    sip_counts = filtered["sip_evidence"].value_counts().rename_axis("Label").reset_index(name="Records")
    st.dataframe(sip_counts, use_container_width=True, hide_index=True)
    st.warning("Counts describe this collected sample only. They are not population estimates, retention rates, or proof of causation. Reviews may be about trading or app performance rather than mutual fund SIPs.")
with tab3:
    st.subheader("Evidence quality and validation")
    st.markdown("**Current limitation:** the candidate theme and SIP labels are generated with transparent keyword rules. They are not a validated LLM workflow yet.")
    st.markdown("**Human validation workflow**")
    st.markdown("1. Export a sample of records.\n2. Independently label each record's main theme and whether it contains direct SIP continuation or stopping evidence.\n3. Compare human labels with the engine labels.\n4. Record disagreements, correct the taxonomy, and report agreement only after labels are completed.")
    if "human_verified" in df.columns:
        verified = df["human_verified"].astype(str).str.lower().isin(["true", "1", "yes"])
        st.metric("Records marked human-verified in source file", int(verified.sum()))
    st.markdown("**Source coverage in current dataset**")
    source_table = df.groupby(["source", "record_type"], dropna=False).size().reset_index(name="Records")
    st.dataframe(source_table, use_container_width=True, hide_index=True)
    st.markdown("**Next data collection gaps**")
    st.write("- Reddit discussions: the earlier collection attempt was blocked or returned no data.\n- YouTube: current entries are video leads, not comment text.\n- Direct SIP behaviour: public app reviews rarely establish whether a person continued a SIP for 12 months. Treat this as a research gap, not a variable we can infer.")


with tab4:
    st.subheader("SIP continuation research")
    st.write(
        "A separate evidence collection for exploring possible reasons investors pause, "
        "stop, resume, or continue SIPs. This starter dataset contains preliminary source leads "
        "and official product documentation, not a representative sample."
    )

    if SIP_RESEARCH_PATH.exists():
        sip_df = pd.read_csv(SIP_RESEARCH_PATH).fillna("")
        expected_cols = [
            "source_id", "source_type", "source_url", "evidence_text",
            "investor_status", "stated_reason", "theme_candidate",
            "groww_mentioned", "evidence_type", "verification_status", "use_in_project"
        ]
        for col in expected_cols:
            if col not in sip_df.columns:
                sip_df[col] = ""

        m1, m2, m3 = st.columns(3)
        m1.metric("Research records", len(sip_df))
        m2.metric("Reddit leads", int(sip_df["source_type"].astype(str).str.contains("Reddit", case=False).sum()))
        m3.metric("Official Groww documents", int(sip_df["source_type"].astype(str).str.contains("Official Groww", case=False).sum()))

        source_choices = ["All"] + sorted(sip_df["source_type"].astype(str).unique().tolist())
        selected_source_type = st.selectbox("Research source type", source_choices, key="sip_source_type")
        if selected_source_type != "All":
            sip_df = sip_df[sip_df["source_type"] == selected_source_type]

        theme_choices = ["All"] + sorted(sip_df["theme_candidate"].astype(str).unique().tolist())
        selected_research_theme = st.selectbox("Research theme", theme_choices, key="sip_research_theme")
        if selected_research_theme != "All":
            sip_df = sip_df[sip_df["theme_candidate"] == selected_research_theme]

        groww_filter = st.selectbox(
            "Groww explicitly mentioned?",
            ["All", "Yes", "No", "Not established"],
            key="sip_groww_filter"
        )
        if groww_filter != "All":
            sip_df = sip_df[sip_df["groww_mentioned"].astype(str).str.lower() == groww_filter.lower()]

        st.dataframe(
            sip_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_url": st.column_config.LinkColumn("Source URL", display_text="Open source"),
                "evidence_text": st.column_config.TextColumn("Evidence text", width="large"),
                "use_in_project": st.column_config.TextColumn("Use in project", width="large"),
                "verification_status": st.column_config.TextColumn("Verification status", width="large"),
            }
        )
        st.download_button(
            "Download SIP research CSV",
            sip_df.to_csv(index=False).encode("utf-8"),
            "groww_sip_research_filtered.csv",
            "text/csv",
            key="download_sip_research"
        )
        st.warning(
            "Evidence quality warning: source leads and classifications are preliminary. "
            "Verify original posts and comments before treating them as investor accounts. "
            "Official help pages describe possible failure mechanisms but do not establish how common they are. "
            "This dataset cannot calculate Groww's 12-month SIP retention."
        )
    else:
        st.error(
            "The file groww_sip_research_starter.csv was not found. "
            "Upload it to the same repository folder as app.py."
        )

st.divider()
st.caption("Built for discovery, not investment advice. No specific funds or stocks are recommended. No fear, FOMO, or urgency-based nudges are generated.")
