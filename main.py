# query.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load the FAISS index and dataset
index = faiss.read_index('faiss_index.index')
data = pd.read_pickle('data.pkl')

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def neural_search(query, top_n=5):
    # Encode the query
    query_combined = f"Title: {query} | Columns: example_value | Tags: example_value | Abstract: example_value"
    query_embedding = model.encode([query_combined], convert_to_tensor=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Perform search
    distances, indices = index.search(query_embedding, top_n)
    
    # Retrieve the top N results
    results = data.iloc[indices[0]]
    
    return results

# Example usage
if __name__ == "__main__":
    query = input("Enter a query: ")
    results = neural_search(query)
    print("Top results for query '{}':".format(query))
    print(results)
