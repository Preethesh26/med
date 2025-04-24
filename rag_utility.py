
import os
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

dataset_path = "plants.xlsx"

def load_dataset():
    df = pd.read_excel(dataset_path)
    documents = []
    for _, row in df.iterrows():
        text_parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and isinstance(row[col], (str, int, float))]
        text = "\n".join(text_parts)
        documents.append(Document(page_content=text, metadata={"source": "plants.xlsx"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def create_vector_db(texts):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(texts, embedding)
    return vectordb

texts = load_dataset()
vectordb = create_vector_db(texts)

def answer_question(query):
    docs = vectordb.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found."

    formatted_response = "## 🌿 Medicinal Plant Information\n\n"
    preparation_method = ""
    unique_responses = set()

    for doc in docs:
        content = doc.page_content.strip()
        if content not in unique_responses:
            unique_responses.add(content)
            lines = content.split("\n")
            for line in lines:
                if "Preparation Method" in line:
                    preparation_method += f"- **{line.split(':', 1)[0]}**: {line.split(':', 1)[1].strip()}\n"
                elif ":" in line:
                    key, value = line.split(":", 1)
                    formatted_response += f"- **{key.strip()}**: {value.strip()}\n"
                else:
                    formatted_response += f"- {line.strip()}\n"
            formatted_response += "\n"

    if preparation_method:
        formatted_response += "\n### 🏺 Preparation Method\n" + preparation_method + "\n---\n"

    return formatted_response
