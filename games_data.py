# model_training_knn.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle

# Load data
data = pd.read_csv('games_data.csv')

# Map textual ratings to numerical values
rating_mapping = {
    "Mixed": 1,
    "Positive": 2,
    "Mostly Positive": 3,
    "Very Positive": 4
}
data['rating'] = data['rating'].map(rating_mapping)

# Select features for similarity calculation
features = ['rating', 'price_final', 'win', 'mac', 'linux', 'positive_ratio']

# Drop rows with missing values in these columns (just in case)
data = data.dropna(subset=features)

# Scale the features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Initialize the KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(data[features])

# Save the KNN model and game titles for later use
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

with open('game_titles.pkl', 'wb') as f:
    pickle.dump(data['title'].tolist(), f)

print("KNN model and game titles saved.")
