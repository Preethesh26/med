
import streamlit as st
import pandas as pd
import os
from rag_utility import answer_question

st.set_page_config(page_title="Medicinal Plant RAG System", layout="wide")
plants_df = pd.read_excel("plants.xlsx")

st.markdown("<h1>🌿 Medicinal Plant RAG System</h1>", unsafe_allow_html=True)
st.write("Ask about medicinal plants and their uses.")

query = st.text_input("🔎 **Ask your question about medicinal plants**")

if st.button("Get Answer"):
    if query:
        with st.spinner("🔍 Searching for the best answer..."):
            answer = answer_question(query)
        st.markdown(answer, unsafe_allow_html=True)

        matched_row = plants_df[plants_df['Plant Name'].str.lower() == query.strip().lower()]
        if not matched_row.empty:
            image_path = matched_row.iloc[0].get("Image", None)
            if pd.notna(image_path):
                try:
                    if image_path.startswith("http"):
                        st.image(image_path, caption="🖼️ Medicinal Plant Image", use_column_width=True)
                    else:
                        full_path = os.path.join("images", image_path) if not os.path.exists(image_path) else image_path
                        if os.path.exists(full_path):
                            st.image(full_path, caption="🖼️ Medicinal Plant Image", use_column_width=True)
                        else:
                            st.warning("⚠️ Image not found.")
                except Exception:
                    st.warning("⚠️ Error displaying image.")
            else:
                st.image("images/placeholder.jpg", caption="🖼️ No image provided for this plant.", use_column_width=True)
        else:
            st.warning("⚠️ No matching plant found.")
    else:
        st.warning("⚠️ Please enter a plant name.")
