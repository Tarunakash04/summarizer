# 🚀 Log Summary & Analysis Tool

This is a production-ready AI-powered tool for summarizing and comparing tabular logs across multiple files (Excel, CSV, PDF). Designed for fast local use, it supports both **file uploads** and **entire folders (ZIPs)** and outputs intelligent summaries along with downloadable Excel reports.

## 🔍 Key Features

- 🧠 **AI-Based Summary** using FLAN-T5  
- 🧾 Handles **Excel, CSV, PDF** formats  
- 📂 Accepts **ZIP folders** with multiple logs  
- 🧮 Smart grouping of **primary + secondary columns**  
- 🎨 Color-coded groups with **legend support**  
- 📊 Generates **comparison tables across files**  
- 📥 One-click **Excel export** with multiple sheets  
- ⚙️ Built using **Flask**, **Transformers**, **Pandas**

## 📸 Screenshots

### 🗂 Upload & File Parsing  
![Upload](static/demo_upload.png)

### 🧠 Column Selection  
![Drag Drop](static/demo_columns.png)

### 📊 AI-Powered Summary  
![Summary](static/demo_summary.png)

### 📥 Excel Report  
![Excel](static/demo_excel.png)

## 🏗 Tech Stack

| Layer         | Tools/Packages                                                                 |
|---------------|----------------------------------------------------------------------------------|
| Frontend      | HTML, CSS (custom), Flask Templates                                             |
| Backend       | Flask, Pandas, XlsxWriter, pdfplumber                                           |
| AI/NLP        | HuggingFace Transformers, FLAN-T5, SentenceTransformers                         |
| File Support  | `.xlsx`, `.xls`, `.csv`, `.pdf`, `.zip`                                         |
| Export Format | `.xlsx` with multiple sheets                                                    |

## ⚙️ Setup Instructions

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
