import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

index = faiss.read_index('models/faiss_index1.index')
data = pd.read_pickle('models/data1.pkl')

model = SentenceTransformer('all-MiniLM-L6-v2')

def neural_search(query, top_n=5):
    query_combined = f"Title: {query} | Columns: example_value | Tags: example_value | Abstract: example_value"
    query_embedding = model.encode([query_combined], convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    distances, indices = index.search(query_embedding, top_n)
    
    results = data.iloc[indices[0]]
    
    return results

st.header("Neural Search Engine")
query = st.text_input("Enter a query:")
if st.button("Search"):
    results = neural_search(query)
    results = results.drop(columns=['combined_text'])
    # print(results.columns)
    st.write(results)