import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset
df = pd.read_csv('data/data3.csv')

# Preprocess the text data
df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x.strip('[]').split(';')))
df['combined_text'] = df.apply(
    lambda row: f"Title: {row['Title']} | Columns: {row['Columns']} | Tags: {row['Tags']} | Abstract: {row['Abstract']}", 
    axis=1
)

df['combined_text'] = df['combined_text'].apply(lambda x: x.lower())

# Apply TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Sum the TF-IDF scores for each word
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))

# Sort the words by their TF-IDF scores
sorted_tfidf_scores = dict(sorted(tfidf_scores_dict.items(), key=lambda x: x[1], reverse=True))

# Get the top 30 words by TF-IDF score
top_30_words = list(sorted_tfidf_scores)[:10]


print(sorted_tfidf_scores['dataset'])

# # Calculate the frequency of these top 30 words in the dataset
# word_list = df['combined_text'].str.split().sum()
# word_freq = Counter(word_list)

# # Filter the frequencies to include only the top 30 words
# top_30_word_freq = {word: freq for word, freq in word_freq.items() if word in top_30_words}

# # Sort the frequencies
# sorted_top_30_word_freq = dict(sorted(top_30_word_freq.items(), key=lambda x: x[1], reverse=True))

# # Plot the most frequent words among the top 30 TF-IDF words
# plt.figure(figsize=(10, 6))
# plt.bar(sorted_top_30_word_freq.keys(), sorted_top_30_word_freq.values())
# plt.xticks(rotation=45)
# plt.title('Most Frequent Words Among Top 7 TF-IDF Words')
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.show()