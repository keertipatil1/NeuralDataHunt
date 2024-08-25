# Draw a bar chart of the most frequent tags in df['Tags']

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/data3.csv')

# Preprocess the text data
df['Tags'] = df['Tags'].apply(lambda x: ', '.join(x.strip('[]').split(';')))

# Get the list of tags
tag_list = df['Tags'].str.split(',').sum()

# Count the frequency of each tag
tag_freq = Counter(tag_list)

# Get the top 10 most frequent tags
top_10_tags = dict(tag_freq.most_common(10))

# Plot the most frequent tags
plt.figure(figsize=(10, 6))
plt.bar(top_10_tags.keys(), top_10_tags.values())
plt.xticks(rotation=45)
plt.title('Most Frequent Tags')
plt.xlabel('Tags')
plt.ylabel('Frequency')

plt.show()