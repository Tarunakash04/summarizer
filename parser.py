import pandas as pd
import pdfplumber
import os

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
    dfs = []
    for path in filepaths:
        if path.endswith(".csv"):
            dfs.append(pd.read_csv(path))
        elif path.endswith(".pdf"):
            dfs.extend(extract_table_from_pdf(path))
    return dfs
