import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data/data3.csv')

df['Columns'] = df['Columns'].apply(lambda x: ', '.join(x.strip('[]').split(';')))
df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x.strip('[]').split(';')))
df['combined_text'] = df.apply(
    lambda row: f"Title: {row['Title']} | Columns: {row['Columns']} | Tags: {row['Tags']} | Abstract: {row['Abstract']}", 
    axis=1
)

df['combined_text'] = df['combined_text'].apply(lambda x: x.lower())

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=False)

embeddings = np.array(embeddings, dtype=np.float32)

tsne = TSNE(n_components=2, random_state=0)

embeddings_3d = tsne.fit_transform(embeddings)

df['x'] = embeddings_3d[:, 0]
df['y'] = embeddings_3d[:, 1]

colors = df['Truth'].unique().tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['x'], df['y'], c=df['Truth'].astype('category').cat.codes, cmap='hsv')

legend1 = ax.legend(*scatter.legend_elements(), title="Truth")
ax.add_artist(legend1)

plt.show()