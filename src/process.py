# preprocess.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset
data = pd.read_csv('data/data.csv')  # Replace with your dataset path

# Split Columns and Tags into separate entries
data['Columns'] = data['Columns'].apply(lambda x: ', '.join(x.strip('[]').split(';')))
data['Tags'] = data['Tags'].apply(lambda x: ', '.join(x.strip('[]').split(';')))

# Combine columns into a single text field
data['combined_text'] = data.apply(
    lambda row: f"Title: {row['Title']} | Columns: {row['Columns']} | Tags: {row['Tags']} | Abstract: {row['Abstract']}", 
    axis=1
)

# Generate embeddings for the combined text
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['combined_text'].tolist(), convert_to_tensor=False)

# Convert embeddings to float32 for FAISS
embeddings = np.array(embeddings, dtype=np.float32)

# Create a FAISS index
dimension = embeddings.shape[1]  # This is the size of the embedding vectors
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for simplicity

# Add embeddings to the index
index.add(embeddings)

# Save the index and data
faiss.write_index(index, 'faiss_index.index')
data.to_pickle('data.pkl')
