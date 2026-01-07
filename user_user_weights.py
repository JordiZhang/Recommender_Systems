from scipy.sparse import coo_array, csr_array
import scipy
import pandas as pd
import numpy as np

movielens = pd.read_csv("movielens20/rating.csv")
movielens = movielens.drop("timestamp", axis=1)

# dictionaries to quickly find the index for the user_movie matrix or vice versa
index_to_user = movielens["userId"].drop_duplicates().sort_values(ascending=True).reset_index().drop("index", axis=1).to_dict()["userId"]
index_to_movie = movielens["movieId"].drop_duplicates().sort_values(ascending=True).reset_index().drop("index", axis=1).to_dict()["movieId"]

user_to_index = dict((v, k) for k, v in index_to_user.items())
movie_to_index = dict((v, k) for k, v in index_to_movie.items())

# we will store user_movie matrix as a sparse matrix to save on memory
ratings = movielens["rating"].to_numpy()
row = movielens["userId"].to_numpy()
col = movielens["movieId"].to_numpy()

# convert user/movie Id into index for sparse matrix
for i in range(len(row)):
    row[i] = user_to_index[row[i]]
    col[i] = movie_to_index[col[i]]

# fast construct using coo, then convert to csr for fast row indexing
user_movie = coo_array((ratings, (row, col)))
user_movie = user_movie.tocsr()



def pearson(x, y, threshold):
    '''
    Takes in the indices of 2 different users and outputs its pearson correlation coefficient. Uses a threshold for
    the minimum number of movies in common to consider a weight.
    '''
    if x == y:
        return 1
    user1 = user_movie[x, :]
    user2 = user_movie[y, :]

    movies1 = user1.coords[0].tolist()
    movies2 = user2.coords[0].tolist()
    combined = movies1 + movies2

    # find movies in common
    u, c = np.unique(combined, return_counts=True)
    common = u[c > 1]
    if len(common) < threshold:
        return 0

    index1 = np.isin(movies1, common)
    index2 = np.isin(movies2, common)

    rating1 = user1.data[index1]
    rating2 = user2.data[index2]

    return scipy.stats.pearsonr(rating1, rating2).statistic



weight = []
row = []
col = []
min_in_common = 5

for i in range(user_movie.shape[0]):
    for j in range(user_movie.shape[0]):
        print("Users: ", i, j)
        cor = pearson(i, j, min_in_common)
        if cor != 0:
            weight.append(cor)
            row.append(i)
            col.append(j)

correlation = coo_array((weight, (row, col)), shape=(user_movie.shape[0], user_movie.shape[0]))
print(correlation)
scipy.sparse.save_npz("pearson_correlations.npz", correlation)




