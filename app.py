from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import pdfplumber
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret_key'

# === Load Models ===
try:
    print("🔁 Loading models...")
    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    print("✅ Models loaded.")
except Exception as e:
    print("❌ Model loading failed:", e)
    sys.exit(1)

HEADER_KEYWORDS = ['log source', 'event name', 'event count', 'start time', 'hostname']

def find_real_header(df_or_rows):
    for i, row in enumerate(df_or_rows[:10]):
        if any(
            any(k.lower() in str(cell).lower() for k in HEADER_KEYWORDS)
            for cell in row
        ):
            return i
    return 0

def extract_table_from_pdf(filepath):
    dfs = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            raw_table = page.extract_table()
            if raw_table and len(raw_table) > 1:
                header_idx = find_real_header(raw_table)
                header = raw_table[header_idx]
                data = raw_table[header_idx + 1:]
                df = pd.DataFrame(data, columns=header)
                dfs.append(df)
    return dfs

def parse_files(filepaths):
    dataframes = []
    for path in filepaths:
        try:
            if path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb", ".ods")):
                temp_df = pd.read_excel(path, header=None)
                header_idx = find_real_header(temp_df.values.tolist())
                df = pd.read_excel(path, header=header_idx)
                dataframes.append(df)
            elif path.endswith(".csv"):
                temp_df = pd.read_csv(path, header=None)
                header_idx = find_real_header(temp_df.values.tolist())
                df = pd.read_csv(path, header=header_idx)
                dataframes.append(df)
            elif path.endswith(".pdf"):
                pdf_dfs = extract_table_from_pdf(path)
                dataframes.extend(pdf_dfs)
        except Exception as e:
            print(f"❌ Parsing failed for {path}: {e}")
    return dataframes

def detect_columns(df_list):
    sets = [set(df.columns) for df in df_list if not df.empty]
    if not sets:
        return [], []
    common = set.intersection(*sets)
    all_cols = set.union(*sets)
    uncommon = all_cols - common
    return sorted(common), sorted(uncommon)

def generate_summary_prompt(df, target, features):
    features = list(features) if isinstance(features, (list, tuple)) else [features]
    sample = df[[target] + features].dropna(subset=[target]).drop_duplicates().head(30)
    if sample.empty:
        sample = df.head(30)

    rows_description = sample.to_dict(orient="records")
    example_facts = "\n".join([f"- {target}: {row.get(target)}, Features: " +
                               ", ".join([f"{f}: {row.get(f)}" for f in features if f in row])
                               for row in rows_description])

    prompt = f"""
You are a professional data analyst. Summarize the following tabular data to highlight meaningful trends, correlations, and insights.

Data characteristics:
- Target column: {target}
- Feature columns: {', '.join(features)}

Sample data (limited to 30 rows):
{example_facts}

Instructions:
- Provide a business-friendly summary, such as "Modules with type X had 29% failure rate."
- Use approximate statistics where needed.
- Include bullet points or concise insights.
- Ensure the output is easy to understand by non-technical stakeholders.
"""
    return prompt

def generate_summary(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = qa_model.generate(**inputs, max_new_tokens=300)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if text.strip().lower().startswith("pass sync") or len(set(text.strip().split())) < 10:
        return "⚠️ The model could not generate a meaningful summary. Try different primary/secondary columns."
    return text

def generate_comparison_table(dataframes, filenames):
    records = []
    for idx, df in enumerate(dataframes):
        if df.empty:
            continue
        record = {
            "Filename": filenames[idx],
            "Rows": len(df),
            "Columns": len(df.columns),
        }
        for col in df.columns:
            try:
                record[f"Unique {col}"] = df[col].nunique()
            except Exception:
                record[f"Unique {col}"] = "N/A"
        records.append(record)
    return pd.DataFrame(records).fillna("N/A") if records else None

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
    selected_secondary_columns = request.form.getlist('secondary_columns')

    if len(selected_secondary_columns) == 1 and "," in selected_secondary_columns[0]:
        selected_secondary_columns = [col.strip() for col in selected_secondary_columns[0].split(",")]

    if not target or not selected_secondary_columns:
        return "Primary and secondary columns must be selected.", 400

    paths = session.get('paths', [])
    dataframes = parse_files(paths)

    selected_data = []
    for df in dataframes:
        if target not in df.columns:
            continue
        usable_features = [col for col in selected_secondary_columns if col in df.columns]
        if not usable_features:
            continue

        cols_to_use = usable_features + [target]
        filtered = df[cols_to_use].dropna(subset=[target])

        if not filtered.empty:
            selected_data.append(filtered)

    if not selected_data:
        all_data = parse_files(paths)
        merged_df = pd.concat(all_data, ignore_index=True)
        usable = [col for col in merged_df.columns if col != target]
        prompt = generate_summary_prompt(merged_df, target, usable)
        model_summary_text = generate_summary(prompt)
        return render_template("summary.html",
                               target=target,
                               summary_table=None,
                               model_summary_text=model_summary_text,
                               supporting_table_html=None,
                               comparison_table_html=None)

    merged_df = pd.concat(selected_data, ignore_index=True)
    session['data'] = merged_df.to_json()
    session['target'] = target
    session['features'] = selected_secondary_columns

    valid_features = [f for f in selected_secondary_columns if f in merged_df.columns]
    prompt = generate_summary_prompt(merged_df, target, valid_features)
    model_summary_text = generate_summary(prompt)

    try:
        merged_df[[target] + valid_features] = merged_df[[target] + valid_features].astype(str)
        supporting_table = (
            merged_df[[target] + valid_features]
            .dropna()
            .groupby([target] + valid_features)
            .size()
            .reset_index(name='Count')
        )
        supporting_table_html = supporting_table.to_html(index=False, classes="summary-table")
    except Exception as e:
        supporting_table_html = None

    # Comparison table if multiple files
    if len(paths) > 1:
        all_data = parse_files(paths)
        comparison_df = generate_comparison_table(all_data, [os.path.basename(p) for p in paths])
        comparison_table_html = comparison_df.to_html(index=False, classes="summary-table") if comparison_df is not None else None
    else:
        comparison_table_html = None

    return render_template("summary.html",
                           target=target,
                           summary_table=None,
                           model_summary_text=model_summary_text,
                           supporting_table_html=supporting_table_html,
                           comparison_table_html=comparison_table_html)

@app.route('/download')
def download():
    # Download Excel with AI summary + supporting table + comparison (if available)
    import xlsxwriter

    json_data = session.get('data', None)
    target = session.get('target', None)
    features = session.get('features', [])

    if json_data is None or target is None:
        return redirect(url_for('index'))

    df = pd.read_json(json_data)

    # Regenerate summary and tables (for safety)
    valid_features = [f for f in features if f in df.columns]
    prompt = generate_summary_prompt(df, target, valid_features)
    summary_text = generate_summary(prompt)

    df[[target] + valid_features] = df[[target] + valid_features].astype(str)
    supporting_table = (
        df[[target] + valid_features]
        .dropna()
        .groupby([target] + valid_features)
        .size()
        .reset_index(name='Count')
    )

    # Multi-file comparison
    paths = session.get('paths', [])
    comparison_df = generate_comparison_table(all_data, [os.path.basename(p) for p in paths])
    if len(paths) > 1:
        all_data = parse_files(paths)
        comparison_df = generate_comparison_table(all_data)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: AI Summary text as a single cell
        summary_df = pd.DataFrame({'AI Summary': [summary_text]})
        summary_df.to_excel(writer, sheet_name='AI_Summary', index=False)

        # Sheet 2: Supporting grouped table
        supporting_table.to_excel(writer, sheet_name='Supporting_Table', index=False)

        # Sheet 3: Comparison table if available
        if comparison_df is not None:
            comparison_df.to_excel(writer, sheet_name='Comparison_Table', index=False)

        writer.save()

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name='summary_export.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/reset')
def reset():
    session.clear()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=False)
