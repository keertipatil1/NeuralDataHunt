import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

def plot_top_tfidf_words(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

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
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores_dict = dict(zip(feature_names, tfidf_scores))
    sorted_tfidf_scores = dict(sorted(tfidf_scores_dict.items(), key=lambda x: x[1], reverse=True))
    top_30_words = list(sorted_tfidf_scores.keys())[:30]
    word_list = df['combined_text'].str.split().sum()
    word_freq = Counter(word_list)
    top_30_word_freq = {word: freq for word, freq in word_freq.items() if word in top_30_words}
    sorted_top_30_word_freq = dict(sorted(top_30_word_freq.items(), key=lambda x: x[1], reverse=True))

    # Plot the most frequent words among the top 30 TF-IDF words
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_top_30_word_freq.keys(), sorted_top_30_word_freq.values())
    plt.xticks(rotation=45)
    plt.title('Most Frequent Words Among Top 30 TF-IDF Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()