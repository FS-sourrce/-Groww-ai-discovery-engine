import re
from pathlib import Path
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Groww AI Discovery Engine", page_icon="🔎", layout="wide")
DATA_PATH = Path(__file__).parent / "research_dataset.csv"

STOPWORDS = {"why","what","when","where","does","after","with","from","that","they","them","their","into","about","have","been","are","the","and","for","was","were","will","would","could","should","this","these","those","you","your","can","might","people"}

def normalize_frame(df, default_source="Research dataset"):
    df = df.copy().fillna("")
    aliases = {"comment":"text", "comment_text":"text", "text_original":"text", "video_title":"title", "published_at":"date", "publishedat":"date", "video_url":"url", "channel_title":"author", "channel":"author"}
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: aliases.get(c,c) for c in df.columns})
    for col, default in {"source":default_source,"record_type":"Evidence","title":"","text":"","url":"","date":"","author":"","rating":""}.items():
        if col not in df.columns: df[col] = default
    df["source"] = df["source"].replace("", default_source)
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().ne("")].copy()

    # Exclude old discovery placeholders that are not actual evidence.
    title_lower = df["title"].str.lower()
    text_lower = df["text"].str.lower()
    placeholder_title = title_lower.str.contains(
        r"video lead for:|youtube video lead|comments on .* not collected", regex=True, na=False
    )
    placeholder_text = text_lower.str.contains(
        r"comments were not collected|comment data were not provided|video discovered through search|comments not collected|content is unavailable",
        regex=True, na=False
    )
    df = df[~(placeholder_title | placeholder_text)].copy()
    return df

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["source","record_type","title","text","url","date","author","rating"])
    return normalize_frame(pd.read_csv(DATA_PATH), "Research dataset")

def search_web(query, api_key, max_results=6):
    response = requests.post("https://api.tavily.com/search", json={
        "api_key":api_key, "query":query + " (Groww OR SIP OR mutual fund investor)",
        "topic":"general", "search_depth":"advanced", "max_results":max_results,
        "include_answer":False, "include_raw_content":True
    }, timeout=45)
    response.raise_for_status()
    rows=[]
    for item in response.json().get("results",[]):
        body=str(item.get("raw_content") or item.get("content") or "").strip()
        if body:
            rows.append({"source":"Live web search", "record_type":"Web page", "title":item.get("title","Untitled page"), "text":body[:4000], "url":item.get("url",""), "date":item.get("published_date","") or "", "author":item.get("author","") or "", "rating":""})
    return pd.DataFrame(rows)

def search_youtube(query, api_key, limit=5):
    response=requests.get("https://www.googleapis.com/youtube/v3/search", params={"part":"snippet","q":query,"type":"video","maxResults":limit,"order":"relevance","regionCode":"IN","relevanceLanguage":"en","key":api_key}, timeout=30)
    response.raise_for_status()
    rows=[]
    for item in response.json().get("items",[]):
        vid=item.get("id",{}).get("videoId","")
        sn=item.get("snippet",{})
        if vid:
            rows.append({"video_id":vid,"title":sn.get("title",""),"channel":sn.get("channelTitle",""),"published_at":sn.get("publishedAt",""),"url":f"https://www.youtube.com/watch?v={vid}","description":sn.get("description","")})
    return pd.DataFrame(rows)

def get_youtube_comments(videos, api_key, per_video=30):
    rows=[]; errors=[]
    for _, video in videos.iterrows():
        params={"part":"snippet","videoId":video["video_id"],"maxResults":min(per_video,100),"order":"relevance","textFormat":"plainText","key":api_key}
        try:
            r=requests.get("https://www.googleapis.com/youtube/v3/commentThreads",params=params,timeout=30)
            if r.status_code != 200:
                errors.append(f"{video['title']}: comments unavailable ({r.status_code})")
                continue
            for item in r.json().get("items",[]):
                sn=item.get("snippet",{}).get("topLevelComment",{}).get("snippet",{})
                text=str(sn.get("textDisplay","")).strip()
                if text:
                    rows.append({"source":"YouTube comments","record_type":"Public YouTube comment","title":video["title"],"text":text,"url":video["url"],"date":sn.get("publishedAt",""),"author":sn.get("authorDisplayName",""),"rating":"","comment_id":item.get("snippet",{}).get("topLevelComment",{}).get("id","")})
        except Exception as exc:
            errors.append(f"{video['title']}: {exc}")
    return pd.DataFrame(rows), errors

def retrieve(question, data, top_k=10):
    data=data.copy()
    data=data[data["text"].astype(str).str.strip().ne("")]
    # Video-only discovery rows are not evidence; comments are evidence.
    data=data[~data["record_type"].astype(str).str.lower().str.contains("video lead|youtube video",regex=True)]
    if data.empty: return data
    corpus=(data["title"].astype(str)+"\n"+data["text"].astype(str)).tolist()
    try:
        vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_features=30000,sublinear_tf=True)
        mat=vec.fit_transform(corpus+[question])
        scores=cosine_similarity(mat[-1],mat[:-1]).ravel()
        idx=[i for i in scores.argsort()[::-1] if scores[i]>0][:top_k]
        out=data.iloc[idx].copy()
        out["relevance"]= [round(float(scores[i]),3) for i in idx]
        return out
    except ValueError:
        return data.head(0)

st.markdown("<h1>Groww AI Discovery Engine</h1><p>Ask one question. Search available evidence. Get an answer with traceable sources.</p>", unsafe_allow_html=True)
st.caption("The engine searches the configured public-web provider and your stored research data. YouTube comments are fetched for relevant videos when the YouTube API key is configured. It cannot guarantee coverage of every public website.")

question=st.text_input("Your research question", placeholder="Why might first-time investors pause or stop their SIPs?")
run=st.button("Search and generate answer", type="primary", use_container_width=True)

if run:
    if not question.strip():
        st.warning("Enter a research question first.")
        st.stop()
    try: secrets=st.secrets
    except Exception: secrets={}
    tavily_key=secrets.get("TAVILY_API_KEY","") if hasattr(secrets,"get") else ""
    youtube_key=secrets.get("YOUTUBE_API_KEY","") if hasattr(secrets,"get") else ""
    groq_key=secrets.get("GROQ_API_KEY","") if hasattr(secrets,"get") else ""
    parts=[]; source_status=[]
    with st.spinner("Searching your dataset and available public sources..."):
        base=load_dataset()
        if not base.empty:
            parts.append(base); source_status.append(("Stored research dataset",len(base),"Searched"))
        else: source_status.append(("Stored research dataset",0,"No usable records"))
        if tavily_key:
            try:
                web_df=search_web(question.strip(),tavily_key,6)
                if not web_df.empty: parts.append(web_df)
                source_status.append(("Live web search",len(web_df),"Searched"))
            except Exception as exc:
                source_status.append(("Live web search",0,f"Search failed: {exc}"))
        else: source_status.append(("Live web search",0,"Not configured: add TAVILY_API_KEY"))
        if youtube_key:
            try:
                videos=search_youtube(question.strip()+" Groww India",youtube_key,5)
                if not videos.empty:
                    comments, comment_errors=get_youtube_comments(videos,youtube_key,30)
                    if not comments.empty: parts.append(comments)
                    source_status.append(("YouTube public comments",len(comments),f"Searched {len(videos)} videos; fetched available comments"))
                    if comment_errors:
                        source_status.append(("YouTube comment access",0,"Some videos had comments disabled or unavailable"))
                else: source_status.append(("YouTube public comments",0,"No relevant videos found"))
            except Exception as exc:
                source_status.append(("YouTube public comments",0,f"Search failed: {exc}"))
        else: source_status.append(("YouTube public comments",0,"Not configured: add YOUTUBE_API_KEY"))
    all_data=pd.concat(parts,ignore_index=True,sort=False).fillna("") if parts else pd.DataFrame()
    if not all_data.empty:
        # Never count placeholder records as evidence, even if their record_type is mislabeled.
        title_lower = all_data["title"].astype(str).str.lower()
        text_lower = all_data["text"].astype(str).str.lower()
        fake_title = title_lower.str.contains(r"video lead for:|youtube video lead", regex=True, na=False)
        fake_text = text_lower.str.contains(
            r"comments were not collected|comment data were not provided|video discovered through search|comments not collected|content is unavailable",
            regex=True, na=False
        )
        all_data = all_data[~(fake_title | fake_text)].copy()
        all_data=all_data.drop_duplicates(subset=[c for c in ["source","url","date","text"] if c in all_data.columns],keep="first")
    evidence=retrieve(question.strip(),all_data,10) if not all_data.empty else pd.DataFrame()
    st.session_state["last_source_status"]=source_status
    st.session_state["last_evidence"]=evidence
    with st.expander("Search status: what was actually collected", expanded=True):
        st.dataframe(
            pd.DataFrame(source_status, columns=["Source","Records collected","Status"]),
            use_container_width=True, hide_index=True
        )
    if evidence.empty:
        st.warning("No relevant evidence was retrieved. Check the source configuration or try a more specific question.")
    else:
        st.subheader("AI research answer")
        if not groq_key:
            st.warning("Evidence was retrieved, but GROQ_API_KEY is not configured, so a generated answer cannot be produced.")
        else:
            contexts=[]
            for n,(_,row) in enumerate(evidence.iterrows(),1):
                contexts.append(f"[{n}] Source type: {row.get('source','')} | Title: {row.get('title','')} | URL: {row.get('url','')}\nEvidence: {str(row.get('text',''))[:2200]}")
            prompt=("Answer the research question using ONLY the supplied evidence. Combine evidence from all source types. Distinguish direct statements from interpretation, identify recurring themes only when supported by multiple records, do not invent prevalence or causation, and explicitly note source gaps. Cite claims inline as [1], [2], etc. Keep the answer structured with: Key findings, What the evidence says, Conflicting/weak evidence, and Limitations.\n\nQuestion: "+question+"\n\nEvidence:\n"+"\n\n".join(contexts))
            try:
                with st.spinner("Analysing evidence from the retrieved sources..."):
                    resp=requests.post("https://api.groq.com/openai/v1/chat/completions",headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"},json={"model":"openai/gpt-oss-120b","messages":[{"role":"system","content":"You are a careful product research analyst. Use only supplied evidence and never fabricate sources, statistics, or citations."},{"role":"user","content":prompt}],"temperature":0.2},timeout=60)
                    resp.raise_for_status()
                    st.markdown(resp.json()["choices"][0]["message"]["content"])
            except Exception as exc:
                st.error(f"AI answer generation failed: {exc}")
        st.subheader("Where the answer's evidence came from")
        source_counts=evidence["source"].value_counts().rename_axis("Source type").reset_index(name="Evidence records")
        st.dataframe(source_counts,use_container_width=True,hide_index=True)
        st.subheader("Sources used")
        show=evidence[[c for c in ["source","title","text","url","date","relevance"] if c in evidence.columns]].copy()
        st.dataframe(show.rename(columns={"source":"Source type","title":"Title","text":"Evidence excerpt","url":"Original source","date":"Date","relevance":"Relevance"}),use_container_width=True,hide_index=True,column_config={"Original source":st.column_config.LinkColumn("Original source",display_text="Open source"),"Evidence excerpt":st.column_config.TextColumn(width="large")})
        st.download_button("Download evidence used (CSV)",evidence.to_csv(index=False).encode("utf-8-sig"),"groww_research_evidence.csv","text/csv")
        st.caption("YouTube comments are retrieved from a small set of relevant videos per search, where comments are available. Search results are not exhaustive, and public comments/reviews cannot prove actual SIP retention or causation.")

with st.expander("Source configuration and limitations"):
    st.write("Required Streamlit Secrets: TAVILY_API_KEY, YOUTUBE_API_KEY, and GROQ_API_KEY. Stored App Store reviews are searched from research_dataset.csv. The app searches a bounded set of web results and YouTube videos for each question, not every page on the internet.")
    if st.button("Show source configuration status"):
        try: sec=st.secrets
        except Exception: sec={}
        st.write({k:("Configured" if sec.get(k,"") else "Missing") for k in ["TAVILY_API_KEY","YOUTUBE_API_KEY","GROQ_API_KEY"]})
