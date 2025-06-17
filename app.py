from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pdfplumber
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from werkzeug.utils import secure_filename
import json
from sentence_transformers import SentenceTransformer
import re
import markdown2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def convert_markdown_tables(md_text):
    pattern = r"(\|.+\|\n\|[-\s|]+\|\n(?:\|.*\|\n?)+)"
    tables = re.findall(pattern, md_text)
    for tbl in tables:
        html_table = markdown2.markdown(tbl)
        md_text = md_text.replace(tbl, html_table)
    return md_text

def extract_table_from_pdf(filepath):
    dfs = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table[1:], columns=table[0])
                dfs.append(df)
    return dfs

def parse_files(filepaths):
    dataframes = []
    for path in filepaths:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            dataframes.append(df)
        elif path.endswith(".pdf"):
            pdf_dfs = extract_table_from_pdf(path)
            dataframes.extend(pdf_dfs)
    return dataframes

def detect_columns(df_list):
    sets = [set(df.columns) for df in df_list if not df.empty]
    if not sets:
        return [], []
    common = set.intersection(*sets)
    all_cols = set.union(*sets)
    uncommon = all_cols - common
    return sorted(common), sorted(uncommon)

def generate_summary_prompt(dataframe, target, features):
    df = dataframe[[*features, target]].dropna().head(10)
    table_text = df.to_csv(index=False)
    prompt = f"""
You are a data analyst. You are given a table of data related to a system's testing logs.

The column '{target}' is the metric of interest. The other columns {', '.join(features)} are feature attributes like configurations, versions, or environment details.

Look at correlations, trends, or rules in the first 10 rows of the table, and explain how these features seem to affect the target.

TABLE:
{table_text}

Give a concise analytical summary.
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = qa_model.generate(**inputs, max_new_tokens=200)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        saved_paths = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            saved_paths.append(save_path)

        dataframes = parse_files(saved_paths)
        common_cols, uncommon_cols = detect_columns(dataframes)
        session['paths'] = saved_paths

        return render_template('drag_drop.html',
                               common_cols=common_cols,
                               uncommon_cols=uncommon_cols)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    target = request.form.get('primary_column')
    features = request.form.getlist('secondary_columns')

    if not target or not features:
        return "Primary and secondary columns must be selected.", 400

    print(f"[DEBUG] Target column: {target}")
    print(f"[DEBUG] Secondary columns: {features}")

    paths = session.get('paths', [])
    dataframes = parse_files(paths)
    print(f"[DEBUG] Number of dataframes parsed: {len(dataframes)}")

    selected_data = []

    for idx, df in enumerate(dataframes):
        print(f"\n[DEBUG] DataFrame #{idx} columns: {list(df.columns)}")

        if target not in df.columns:
            print(f"[DEBUG] Skipping DF #{idx} because target '{target}' not found")
            continue

        usable_features = [col for col in features if col in df.columns]
        print(f"[DEBUG] Usable secondary features in DF #{idx}: {usable_features}")

        if not usable_features:
            print(f"[DEBUG] Skipping DF #{idx} because no secondary features found")
            continue

        cols_to_use = usable_features + [target]
        filtered = df[cols_to_use].dropna(subset=[target])

        print(f"[DEBUG] Filtered DF #{idx} has {len(filtered)} rows")

        if not filtered.empty:
            selected_data.append(filtered)

    print(f"\n[DEBUG] Total selected non-empty dataframes: {len(selected_data)}")

    if not selected_data:
        return "No data found.", 400

    merged_df = pd.concat(selected_data, ignore_index=True)
    print(f"[DEBUG] Merged DataFrame shape: {merged_df.shape}")

    session['data'] = merged_df.to_json()

    summary_table = {}
    for feature in features:
        if feature in merged_df.columns:
            grouped = merged_df.groupby(target)[feature].value_counts().unstack(fill_value=0)
            summary_table[feature] = grouped.to_dict(orient='index')

    session['summary_table'] = summary_table
    session['target'] = target

    return render_template('summary.html',
                           summary_table=summary_table,
                           target=target,
                           features=features)


@app.route('/followup', methods=['POST'])
def followup():
    question = request.form['question']
    try:
        json_data = session.get('data')
        if not json_data:
            return "No data found in session", 400

        df = pd.read_json(json_data)
        sample = df.head(200).to_csv(index=False)
        columns = ", ".join(df.columns)

        prompt = f"""
You are a data analyst.
You have a table with the following columns: {columns}

Here is a sample of the dataset:

{sample}

Now answer this question clearly and analytically: "{question}"

1. If the question asks about frequency or distribution (e.g. "most frequent", "most failed", "highest count"), analyze the entire column.
2. Give a summary in 2-3 sentences.
3. If comparison helps, show a small table using markdown like this:

| Value | Count |
|-------|-------|
| X     | 10    |

Be accurate and specific.
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = qa_model.generate(**inputs, max_new_tokens=200)
        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = convert_markdown_tables(raw_answer)

        return render_template("summary.html",
                               summary_table=session.get("summary_table"),
                               target=session.get("target"),
                               followup_question=question,
                               followup_answer=answer)
    except json.JSONDecodeError as e:
        return f"JSON decode error: {str(e)}", 400
    except Exception as e:
        return f"Error processing follow-up: {str(e)}", 500

@app.route('/reset')
def reset():
    session.clear()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
    return redirect(url_for('index'))

app.secret_key = 'secret_key'

if __name__ == '__main__':
    app.run(debug=True)
