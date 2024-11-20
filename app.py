# app.py
import streamlit as st
import pandas as pd
import pickle

# Load data
with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

with open('game_titles.pkl', 'rb') as f:
    game_titles = pickle.load(f)

# Streamlit UI
st.title("Video Game Recommendation System")
st.write("Find similar games based on your preferences!")

# User input: Select a game
selected_game = st.selectbox("Choose a game you like:", game_titles)

# Get recommendations
if st.button("Get Recommendations"):
    # Find the index of the selected game
    game_idx = game_titles.index(selected_game)
    
    # Get similarity scores for the selected game and sort them
    similarity_scores = list(enumerate(similarity_matrix[game_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Display top 5 similar games
    st.write("Games similar to", selected_game, ":")
    for i, (idx, score) in enumerate(similarity_scores[1:6], 1):
        st.write(f"{i}. {game_titles[idx]} (Similarity: {score:.2f})")
