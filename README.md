This Python script implements a movie recommendation system using a content-based filtering approach. The system leverages the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization method to convert textual features of movies into numerical representations. It then computes cosine similarity scores between movies to generate recommendations based on content similarity.

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df=pd.read_csv('/content/movies.csv')
# Print the first few rows 
print("First few rows of the dataset:\n", df.head())
# Combining important features into a single string
#  to handle missing values
df['important_features'] = df['genre'].fillna('') + ' ' + df['director'].fillna('') + ' ' + df['star'].fillna('')
# Printing the new DataFrame with the combined features for verification
print("Dataset with important features:\n", df[['name', 'important_features']].head())
# Import the necessary class
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert textual data to numerical data using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['important_features'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
from sklearn.metrics.pairwise import linear_kernel
# Computing cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Print the shape of the cosine similarity matrix
print("Cosine similarity matrix shape:", cosine_sim.shape)

def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the movie title is in the dataset
    if title not in df['name'].values:
        return "Movie was not found in the dataset."
    
    # Get the index of the movie that matches the title
    idx = df.index[df['name'] == title].tolist()[0] 

    # Get similarity scores for all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top 5 most similar movies (excluding the first one as it is the movie itself)
    sim_scores = sim_scores[1:6]

    # Extract movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return top 5 similar movies
    return df['name'].iloc[movie_indices].tolist()
  # Example usage: Get recommendations for a specific movie
movie_title = "The Evil Dead"  
print(f"Recommendations for '{movie_title}':\n", get_recommendations(movie_title))
# Test with another movie
test_title = "Das Boot"  # can replace with another title
print(f"Recommendations for '{test_title}':\n", get_recommendations(test_title))


