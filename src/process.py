import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

data = pd.read_csv('data/data3.csv')  

data['Columns'] = data['Columns'].apply(lambda x: ', '.join(x.strip('[]').split(';')))
data['Tags'] = data['Tags'].apply(lambda x: ', '.join(x.strip('[]').split(';')))

data['combined_text'] = data.apply(
    lambda row: f"Title: {row['Title']} | Columns: {row['Columns']} | Tags: {row['Tags']} | Abstract: {row['Abstract']}", 
    axis=1
)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['combined_text'].tolist(), convert_to_tensor=False)

embeddings = np.array(embeddings, dtype=np.float32)

dimension = embeddings.shape[1] 
index = faiss.IndexFlatL2(dimension) 

index.add(embeddings)

faiss.write_index(index, 'models/faiss_index1.index')
data.to_pickle('models/data1.pkl')
