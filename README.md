# Groww AI Discovery Engine MVP

A deployable Streamlit prototype for exploring public feedback about Groww. It includes searchable records, filters, transparent keyword-based candidate themes, SIP-evidence labels, exports, and validation guidance.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Streamlit will provide a local address to open in your browser.

## Deploy free with Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `research_dataset.csv`.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy. Copy the public app URL once it finishes.

Only publish data you are permitted to share publicly. Review the dataset before making the repository public.

## Current limitations

- Themes and SIP labels are heuristic keyword suggestions, not validated LLM outputs.
- Current dataset has 166 records: 106 Apple App Store records, 45 Google Play records, and 15 YouTube video leads. Video leads are not comments.
- The dataset does not establish actual 12-month SIP retention or prove why an investor stopped or continued.
- Reddit coverage is currently missing/blocked. Human validation and real user testing still need to be completed.
- Do not present theme counts as representative of all Groww users.
