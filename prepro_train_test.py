import scipy.sparse
from scipy.sparse import coo_array, csr_array, csc_array, diags
import pandas as pd
import time
import pickle
import numpy as np

t1 = time.time()

movielens = pd.read_csv(r"C:\Users\jordi\Documents\Projects Repos\Recommender Systems\movielens20\rating.csv")
movielens = movielens.drop("timestamp", axis=1)

print("Movielens Data Imported.")
print(time.time()-t1)


N_users = -1
N_movies = -1

top_users = movielens['userId'].value_counts().reset_index().to_numpy()[0:N_users, 0]
top_movies = movielens['movieId'].value_counts().reset_index().to_numpy()[0:N_movies, 0]

movielens = movielens[movielens['userId'].isin(top_users) & movielens['movieId'].isin(top_movies)]

print("Top Users and Movies Selected.")
print(time.time()-t1)

# dictionaries to quickly find the index for the user_movie matrix or vice versa
index_to_user = movielens["userId"].drop_duplicates().sort_values(ascending=True).reset_index().drop("index", axis=1).to_dict()["userId"]
index_to_movie = movielens["movieId"].drop_duplicates().sort_values(ascending=True).reset_index().drop("index", axis=1).to_dict()["movieId"]
with open('processed_20k_4k/index_to_movie.pkl', 'wb') as f:
    pickle.dump(index_to_movie, f)
with open('processed_20k_4k/index_to_user.pkl', 'wb') as f:
    pickle.dump(index_to_user, f)


user_to_index = dict((v, k) for k, v in index_to_user.items())
movie_to_index = dict((v, k) for k, v in index_to_movie.items())

# we will store user_movie matrix as a sparse matrix to save on memory
ratings = movielens["rating"].to_numpy()
user = movielens["userId"].to_numpy()
movie = movielens["movieId"].to_numpy()

# convert user/movie Id into index for sparse matrix
for i in range(len(user)):
    user[i] = user_to_index[user[i]]
    movie[i] = movie_to_index[movie[i]]

# fast construct using coo, then convert to csr for fast row indexing
user_movie = coo_array((ratings, (user, movie)))
user_movie = user_movie.tocsr()

def train_test_split(R, split_ratio = 0.2):
    np.random.seed(48)

    train_user = []
    train_movie = []
    train_rating = []

    test_user = []
    test_movie = []
    test_rating = []

    for user in range(R.shape[0]):
        start = R.indptr[user]
        end = R.indptr[user + 1]

        ratings = R.data[start:end]
        movies = R.indices[start:end]

        # minimum of 1 test case per user
        n_test = max(1, int(split_ratio*len(movies)))
        test_index = np.random.choice(len(movies), size=n_test, replace=False)
        mask = np.ones(len(movies), dtype=bool)
        mask[test_index] = False

        # Train entries
        train_user.extend([user] * np.sum(mask))
        train_movie.extend(movies[mask])
        train_rating.extend(ratings[mask])

        # Test entries
        test_user.extend([user] * n_test)
        test_movie.extend(movies[test_index])
        test_rating.extend(ratings[test_index])

    R_train = coo_array((train_rating, (train_user, train_movie)), shape=R.shape)
    R_test = coo_array((test_rating, (test_user, test_movie)), shape=R.shape)

    return R_train, R_test

train_user_movie, test_user_movie = train_test_split(user_movie)

# save train test split
scipy.sparse.save_npz('processed_20k_4k/train_user_movie.npz', train_user_movie)
scipy.sparse.save_npz('processed_20k_4k/test_user_movie.npz', test_user_movie)

print("Train Test Split Done.")
print(time.time()-t1)