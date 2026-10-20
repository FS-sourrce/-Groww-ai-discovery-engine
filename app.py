import ast
import re
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def extract_video_id(value):
    """Accept a YouTube video ID or common YouTube URL formats."""
    value = str(value).strip()
    if not value:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    patterns = [
        r"(?:youtube\.com/watch\?.*?[&?]v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return ""


def search_youtube_videos(query, api_key, max_results=10):
    """Search public YouTube videos with the official YouTube Data API."""
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max(1, min(int(max_results), 50)),
        "order": "relevance",
        "regionCode": "IN",
        "relevanceLanguage": "en",
        "key": api_key,
    }
    response = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params=params,
        timeout=30,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"YouTube search failed (HTTP {response.status_code}): {detail}")
    payload = response.json()
    videos = []
    for item in payload.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        if not video_id:
            continue
        videos.append({
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "description": snippet.get("description", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        })
    return pd.DataFrame(videos)


def fetch_youtube_comments(video_ids, api_key, max_comments_each):
    """Fetch public top-level comments using the official YouTube Data API."""
    all_rows = []
    errors = []
    for video_id in video_ids:
        page_token = None
        fetched_for_video = 0
        while fetched_for_video < max_comments_each:
            batch_size = min(100, max_comments_each - fetched_for_video)
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": batch_size,
                "order": "time",
                "textFormat": "plainText",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                response = requests.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params=params,
                    timeout=30,
                )
                if response.status_code != 200:
                    try:
                        detail = response.json().get("error", {}).get("message", response.text)
                    except Exception:
                        detail = response.text
                    errors.append(f"{video_id}: HTTP {response.status_code}: {detail}")
                    break
                payload = response.json()
                items = payload.get("items", [])
                if not items:
                    break
                for item in items:
                    snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                    all_rows.append({
                        "source": "YouTube",
                        "record_type": "YouTube comment",
                        "title": snippet.get("videoId", video_id),
                        "text": snippet.get("textDisplay", ""),
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "date": snippet.get("publishedAt", ""),
                        "author": snippet.get("authorDisplayName", ""),
                        "rating": "",
                        "comment_id": item.get("snippet", {}).get("topLevelComment", {}).get("id", ""),
                        "video_id": video_id,
                        "like_count": snippet.get("likeCount", 0),
                        "search_blob": "",
                        "theme": "",
                        "sip_evidence": "",
                    })
                fetched_for_video += len(items)
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
            except Exception as exc:
                errors.append(f"{video_id}: {exc}")
                break
    return pd.DataFrame(all_rows), errors

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load research_dataset.csv: {e}")
    st.stop()

# Keep collected YouTube comments available during this app session and include them in AI search.
if "youtube_comments_df" in st.session_state and not st.session_state["youtube_comments_df"].empty:
    collected = st.session_state["youtube_comments_df"].copy()
    for col in ["source", "record_type", "title", "text", "url", "date", "author", "rating"]:
        if col not in collected.columns:
            collected[col] = ""
    collected["text"] = collected["text"].astype(str)
    collected["search_blob"] = (collected["title"].astype(str) + " " + collected["text"].astype(str)).str.lower()
    collected["theme"] = collected.apply(classify_theme, axis=1)
    collected["sip_evidence"] = collected["text"].apply(classify_sip)
    df = pd.concat([df, collected[df.columns]], ignore_index=True).drop_duplicates(subset=["url", "date", "text"], keep="first")

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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["AI Search", "Feedback Explorer", "Theme Overview", "Evidence & Validation", "SIP Research", "YouTube Comments"])
with tab1:
    st.subheader("Ask the research evidence")
    st.write("Ask a question in plain English. The engine retrieves relevant records from the loaded dataset and, when an API key is configured, drafts an AI answer grounded in those records.")
    question = st.text_input("Your research question", placeholder="Why might first-time investors pause or stop SIPs?", key="ai_search_question")
    top_k = st.slider("Evidence records to retrieve", min_value=3, max_value=10, value=5, key="ai_top_k")
    run_search = st.button("Search evidence", type="primary", key="run_ai_search")
    if run_search:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            corpus = (df["title"].fillna("").astype(str) + "\n" + df["text"].fillna("").astype(str)).tolist()
            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
                matrix = vectorizer.fit_transform(corpus + [question])
                scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
                ranked_idx = scores.argsort()[::-1]
                ranked_idx = [i for i in ranked_idx if scores[i] > 0][:top_k]
            except ValueError:
                ranked_idx = []
            if not ranked_idx:
                st.warning("No relevant evidence was found in the current dataset. Try a more specific or differently worded question.")
            else:
                evidence = df.iloc[ranked_idx].copy()
                evidence["relevance_score"] = [round(float(scores[i]), 3) for i in ranked_idx]
                st.markdown("#### Retrieved evidence")
                show = evidence[["source", "title", "text", "url", "relevance_score"]].copy()
                st.dataframe(show.rename(columns={"source":"Source", "title":"Title", "text":"Evidence", "url":"Source URL", "relevance_score":"Relevance"}), use_container_width=True, hide_index=True, column_config={"Source URL": st.column_config.LinkColumn("Source URL", display_text="Open source"), "Evidence": st.column_config.TextColumn(width="large")})
                try:
                    api_key = st.secrets.get("GROQ_API_KEY", "")
                except Exception:
                    api_key = ""
                if not api_key:
                    api_key = st.text_input("Groq API key (optional, not saved)", type="password", key="groq_key_input", help="Add a Groq API key to generate a written answer. Without it, semantic retrieval still works.")
                if api_key:
                    context_parts = []
                    for n, (_, row) in enumerate(evidence.iterrows(), start=1):
                        context_parts.append(f"[{n}] Source: {row.get('source','')} | Title: {row.get('title','')} | URL: {row.get('url','')}\nEvidence: {str(row.get('text',''))[:2500]}")
                    prompt = ("Answer the user's research question using ONLY the evidence below. Do not invent facts or claim the sample represents all Groww investors. Clearly distinguish direct evidence from cautious interpretation. If evidence is insufficient, say so. Cite evidence inline as [1], [2], etc., matching the numbered records.\n\nQuestion: " + question + "\n\nEvidence:\n" + "\n\n".join(context_parts))
                    try:
                        with st.spinner("Generating an evidence-grounded answer..."):
                            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model":"openai/gpt-oss-120b", "messages":[{"role":"system","content":"You are a careful product research analyst. Never make up evidence, statistics, or citations."},{"role":"user","content":prompt}], "temperature":0.2}, timeout=45)
                        response.raise_for_status()
                        answer = response.json()["choices"][0]["message"]["content"]
                        st.markdown("#### AI answer")
                        st.markdown(answer)
                    except Exception as exc:
                        st.error("AI answer generation failed. Check that the API key is valid and try again. Retrieval results above are still available.")
                        st.caption(f"Technical detail: {exc}")
                else:
                    st.info("Semantic retrieval is working. To generate written AI answers, add a Groq API key in Streamlit Secrets as GROQ_API_KEY, or enter one above for this session.")
                st.caption("Method note: TF-IDF semantic-style retrieval ranks textual similarity; it is not an embedding model. Results are exploratory and must be checked against original sources. Public discussions cannot establish Groww's actual 12-month SIP retention rate.")

with tab2:
    st.subheader("Search results")
    if filtered.empty:
        st.warning("No matching records. Try a broader search or remove a filter.")
    else:
        show_cols = ["source", "record_type", "date", "rating", "theme", "sip_evidence", "title", "text", "url"]
        st.dataframe(filtered[show_cols].rename(columns={"source":"Source", "record_type":"Record type", "date":"Date", "rating":"Rating", "theme":"Candidate theme", "sip_evidence":"SIP evidence label", "title":"Title", "text":"Original feedback", "url":"Source URL"}), use_container_width=True, hide_index=True, column_config={"Source URL": st.column_config.LinkColumn("Source URL", display_text="Open source"), "Original feedback": st.column_config.TextColumn(width="large")})
        st.download_button("Download filtered evidence (CSV)", filtered[show_cols].to_csv(index=False).encode("utf-8"), "groww_filtered_evidence.csv", "text/csv")
        st.caption("Candidate themes and SIP labels are heuristic suggestions, not validated AI findings.")
with tab3:
    st.subheader("What appears in this dataset")
    theme_counts = filtered["theme"].value_counts().rename_axis("Candidate theme").reset_index(name="Records")
    if not theme_counts.empty:
        st.bar_chart(theme_counts.set_index("Candidate theme"))
        st.dataframe(theme_counts, use_container_width=True, hide_index=True)
    st.markdown("#### SIP evidence labels")
    sip_counts = filtered["sip_evidence"].value_counts().rename_axis("Label").reset_index(name="Records")
    st.dataframe(sip_counts, use_container_width=True, hide_index=True)
    st.warning("Counts describe this collected sample only. They are not population estimates, retention rates, or proof of causation. Reviews may be about trading or app performance rather than mutual fund SIPs.")
with tab4:
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


with tab5:
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


with tab6:
    st.subheader("Find videos and collect public YouTube comments")
    st.write("Search for relevant videos, select the ones you want, and collect their public top-level comments.")
    st.caption("Uses the official YouTube Data API. The key is read from Streamlit Secrets as YOUTUBE_API_KEY.")
    try:
        youtube_key = st.secrets.get("YOUTUBE_API_KEY", "")
    except Exception:
        youtube_key = ""
    if not youtube_key:
        st.warning("Add YOUTUBE_API_KEY in Streamlit app settings → Secrets before using YouTube tools.")
        st.code('YOUTUBE_API_KEY = "your_api_key_here"', language="toml")

    query = st.text_input(
        "Search YouTube videos",
        value="Groww SIP mutual fund investor experience",
        key="youtube_search_query",
        placeholder="e.g. why people stop SIP mutual funds India",
    )
    search_limit = st.selectbox("Videos to find", [5, 10, 20, 25], index=1, key="youtube_search_limit")
    if st.button("Search videos", type="primary", key="search_youtube_videos"):
        if not youtube_key:
            st.error("No YOUTUBE_API_KEY found in Streamlit Secrets.")
        elif not query.strip():
            st.error("Enter a search phrase.")
        else:
            try:
                with st.spinner("Searching YouTube..."):
                    found = search_youtube_videos(query.strip(), youtube_key, search_limit)
                st.session_state["youtube_found_videos"] = found
                if found.empty:
                    st.info("No videos found for that query.")
                else:
                    st.success(f"Found {len(found)} videos.")
            except Exception as exc:
                st.error(str(exc))

    found = st.session_state.get("youtube_found_videos", pd.DataFrame())
    selected_video_ids = []
    if not found.empty:
        st.markdown("#### Search results")
        display = found.copy()
        display["select"] = False
        edited = st.data_editor(
            display[["select", "title", "channel", "published_at", "url"]],
            use_container_width=True,
            hide_index=True,
            disabled=["title", "channel", "published_at", "url"],
            column_config={
                "select": st.column_config.CheckboxColumn("Collect comments?", default=False),
                "title": st.column_config.TextColumn("Video title", width="large"),
                "url": st.column_config.LinkColumn("Video link", display_text="Open video"),
            },
            key="youtube_video_selection",
        )
        selected_video_ids = found.loc[edited["select"].fillna(False).values, "video_id"].tolist()

        st.caption("Select one or more videos in the first column, then collect comments.")
        max_comments_each = st.selectbox(
            "Maximum comments per selected video",
            [50, 100, 200, 500],
            index=1,
            key="youtube_max_comments",
        )
        if st.button("Collect comments from selected videos", type="primary", key="collect_selected_youtube"):
            if not selected_video_ids:
                st.error("Select at least one video first.")
            elif not youtube_key:
                st.error("No YOUTUBE_API_KEY found in Streamlit Secrets.")
            else:
                all_comments = []
                all_errors = []
                with st.spinner("Collecting comments..."):
                    for video_id in selected_video_ids:
                        comments, errors = fetch_youtube_comments([video_id], youtube_key, int(max_comments_each))
                        if not comments.empty:
                            all_comments.append(comments)
                        all_errors.extend(errors)
                if all_comments:
                    collected = pd.concat(all_comments, ignore_index=True)
                    old = st.session_state.get("youtube_comments_df", pd.DataFrame())
                    combined = pd.concat([old, collected], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["comment_id"], keep="first")
                    st.session_state["youtube_comments_df"] = combined
                    st.success(f"Collected {len(collected)} comments in this run. {len(combined)} unique comments are available in this session.")
                else:
                    st.warning("No comments collected. Comments may be disabled, or the API may be restricted.")
                for error in all_errors:
                    st.error(error)

    st.markdown("---")
    st.markdown("#### Collected comments")
    comments_now = st.session_state.get("youtube_comments_df", pd.DataFrame())
    if not comments_now.empty:
        st.dataframe(
            comments_now[["date", "author", "text", "url", "like_count"]].rename(
                columns={"date":"Published", "author":"Author", "text":"Comment", "url":"Video", "like_count":"Likes"}
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Video": st.column_config.LinkColumn("Video", display_text="Open video"),
                "Comment": st.column_config.TextColumn(width="large"),
            },
        )
        st.download_button(
            "Download collected YouTube comments CSV",
            comments_now.to_csv(index=False).encode("utf-8-sig"),
            "groww_youtube_comments.csv",
            "text/csv",
            key="download_youtube_comments",
        )
        st.info("Collected comments are also included in AI Search and Feedback Explorer for this app session. Download the CSV before closing the app to preserve the collection.")
    else:
        st.info("No comments collected yet. Search videos above, select relevant results, then collect comments.")
    st.caption("Only public top-level comments are collected. Replies are not included. Search results are leads, not proof of investor retention or churn.")

st.divider()
st.caption("Built for discovery, not investment advice. No specific funds or stocks are recommended. No fear, FOMO, or urgency-based nudges are generated.")
