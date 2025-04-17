from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


data = pd.read_csv('./data/example.csv')  

vectorizer = CountVectorizer(ngram_range=(2, 2))  # This  will find bigrams, adjust as needed1

X = vectorizer.fit_transform(data['message'])

bigram_counts = X.sum(axis=0)

# Map from feature integer indices to feature name (bigram)
bigrams = [(bigram, bigram_counts[0, idx]) for bigram, idx in vectorizer.vocabulary_.items()]

# Sort the bigrams by their count
sorted_bigrams = sorted(bigrams, key=lambda x: x[1], reverse=True)

# Top 10 most common bigrams
print(sorted_bigrams[:10])