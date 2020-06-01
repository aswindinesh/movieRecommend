import pandas as pd
import numpy as np
from omdbapi.movie_search import GetMovie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df1=pd.read_csv('./input/tmdb_5000_credits.csv')
df2=pd.read_csv('./input/tmdb_5000_movies.csv')

tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    title = df2['title'].iloc[movie_indices]
    date = df2['release_date'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','Year','Plot','Ratings','Genre'])
    return_df['Title']=title
    return_df['Year']=date
    plot = []
    genre = []
    ratings = []
    for i in title:
        movie = GetMovie(title=i, api_key='65b3ddf')
        mov = movie.get_data('Genre','Ratings','Plot')
        plot.append(mov['Plot'])
        ratings.append(mov['Ratings'][0]['Value'])
        genre.append(mov['Genre'])
    return_df['Plot']=plot
    return_df['Ratings']=ratings
    return_df['Genre']=genre
    return_df.sort_values(by=['Ratings'], inplace=True, ascending=False)
    return return_df

movie_name = input("enter the movie you watched: ")
print(get_recommendations(movie_name))
