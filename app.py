import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader  # for PDF reading

# === File Upload ===
st.title("📑 Plagiarism Checker")
st.write("Upload multiple `.txt` or `.pdf` files to check similarity.")

uploaded_files = st.file_uploader(
    "Upload text/pdf files", type=["txt", "pdf"], accept_multiple_files=True
)

# === Helper functions ===
def extract_text(file):
    """Extract text from txt or pdf"""
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    return ""

def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity_matrix(vectors):
    return cosine_similarity(vectors)

# === Processing ===
if uploaded_files:
    # Read all uploaded files
    student_files = [file.name for file in uploaded_files]
    student_notes = [extract_text(file) for file in uploaded_files]

    # Convert to vectors
    vectors = vectorize(student_notes)

    # Compute similarity matrix
    sim_matrix = similarity_matrix(vectors)

    # === Display results ===
    st.subheader("Plagiarism Results (Pairwise Similarity)")
    results = []
    for i in range(len(student_files)):
        for j in range(i + 1, len(student_files)):
            results.append((student_files[i], student_files[j], round(sim_matrix[i][j], 3)))

    for pair in sorted(results, key=lambda x: -x[2]):
        st.write(f"**{pair[0]}** vs **{pair[1]}** → Similarity: `{pair[2]}`")

    # === Heatmap Visualization ===
    st.subheader("📊 Similarity Heatmap")
    df_sim = pd.DataFrame(sim_matrix, index=student_files, columns=student_files)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_sim, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
