import streamlit as st
import os
import shutil
import pandas as pd
from parser import parse_files
from utils import detect_columns, construct_summary_prompt
from model import get_summary

st.set_page_config(layout="wide")

UPLOAD_FOLDER = "uploads"
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("📊 Log Insight Analyzer")

uploaded = st.file_uploader("Upload PDFs or CSVs", type=['pdf', 'csv'], accept_multiple_files=True)

if uploaded:
    saved_paths = []
    for file in uploaded:
        path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        saved_paths.append(path)

    dfs = parse_files(saved_paths)
    common, uncommon = detect_columns(dfs)

    st.markdown("### 🧩 Select Columns")
    st.write("**Common Columns:**", common)
    st.write("**Uncommon Columns:**", uncommon)

    all_cols = sorted(set(common + uncommon))

    col1, col2 = st.columns([1, 3])
    with col1:
        target = st.selectbox("🎯 Choose primary (target) column", all_cols)
    with col2:
        features = st.multiselect("📌 Choose secondary (feature) columns", [col for col in all_cols if col != target])

    if st.button("🔍 Analyze"):
        selected_data = []
        for df in dfs:
            try:
                selected_data.append(df[[*features, target]])
            except:
                continue

        if not selected_data:
            st.error("No valid data found.")
        else:
            merged = pd.concat(selected_data, ignore_index=True)
            st.session_state["merged_df"] = merged

            prompt = construct_summary_prompt(merged, target, features)
            with st.spinner("Generating summary..."):
                summary = get_summary(prompt)

            st.markdown("## 📈 Summary Report")
            st.info(summary)

            st.markdown("## 💬 Ask a follow-up question")
            question = st.text_input("Type your question here")
            if st.button("➡️ Ask"):
                sample = merged.head(200).to_csv(index=False)
                q_prompt = f"""
You are a data analyst.

Columns: {', '.join(merged.columns)}

Data sample:
{sample}

Now answer this question analytically: "{question}"
Provide a short summary, and if relevant, add a markdown table.

| Value | Count |
|-------|-------|
"""
                with st.spinner("Thinking..."):
                    answer = get_summary(q_prompt)
                st.success(answer)
