from scipy.sparse import coo_array, csr_array, csc_array, diags
import scipy
import numpy as np
import time

t1 = time.time()

train_user_movie = scipy.sparse.load_npz('../processed/train_user_movie.npz')
test_user_movie = scipy.sparse.load_npz('../processed/test_user_movie.npz')

print("Import train test splits")
print(time.time()-t1)

user_means = np.zeros(train_user_movie.shape[0])

for u in range(train_user_movie.shape[0]):
    row = train_user_movie[u, :]
    if row.nnz > 0:
        user_means[u] = row.data.mean()
    else:
        user_means[u] = 0

print("User Means Calculated")
print(time.time()-t1)

def pearson(R, threshold = 5):
    '''
    Takes in user_movie matrix and returns a pearson correlation matrix.
    '''
    R = R.copy()
    # mean center the user-movie matrix
    for u in range(R.shape[0]):
        start = R.indptr[u]
        end = R.indptr[u + 1]
        if start != end:
            R.data[start:end] -= user_means[u]
    # calculate numerator
    similarity = R @ R.T

    # count movies in common by users and set correlations of those who dont pass threshold to 0
    R_binary = R.copy()
    R_binary.data[:] = 1  # convert to 0/1
    common_counts = R_binary @ R_binary.T
    mask_matrix = common_counts >= threshold
    similarity = similarity.multiply(mask_matrix)

    # find user norms
    row_sums = R.multiply(R).sum(axis=1)
    row_norms = np.sqrt(np.array(row_sums).ravel())

    # invert norms for linalg trick
    inv_norms = np.reciprocal(row_norms, where=row_norms != 0)
    d_inv = diags(inv_norms)

    # combine everything
    corr = d_inv @ similarity @ d_inv
    return corr


correlation = pearson(train_user_movie, threshold=5)
scipy.sparse.save_npz("correlation.npz", correlation)

print("Correlation Matrix Calculated.")
print(time.time()-t1)

# change to csc for column indexing now
train_user_movie = csc_array(train_user_movie)

def predict(x, y, k_neighbours=25):
    '''
    Given user x and movie y, predict the rating.
    '''
    # users who rated movie y
    col = train_user_movie[:, y]
    users_who_rated = col.nonzero()[0]
    print(users_who_rated)

    if len(users_who_rated) == 0:
        return user_means[x]
    sims = correlation[x, users_who_rated].toarray().ravel()

    # remove self
    mask = users_who_rated != x
    users_who_rated = users_who_rated[mask]
    sims = sims[mask]

    # remove zero weights
    nonzero = sims != 0
    users_who_rated = users_who_rated[nonzero]
    sims = sims[nonzero]

    if len(sims) == 0:
        return user_means[x]

    # take top-k among those
    k = min(k_neighbours, len(sims))
    top_k_idx = np.argpartition(np.abs(sims), -k)[-k:]

    selected_users = users_who_rated[top_k_idx]
    selected_sims = sims[top_k_idx]

    ratings = train_user_movie[selected_users, y].toarray().ravel()

    numerator = np.sum(selected_sims * (ratings - user_means[selected_users]))
    denominator = np.sum(np.abs(selected_sims))

    # in case all top 25 neighbours have 0 correlation, i.e. they dont meet threshold
    if denominator == 0:
        return user_means[x]

    return user_means[x] + numerator / denominator

# now do the predictions on test data and find the MSE
predictions = []
rows, cols = test_user_movie.coords
for u, m in zip(rows, cols):
    predictions.append(predict(u, m))

MSE = np.mean((test_user_movie.data - predictions)**2)
print('Prediction MSE:',MSE)
print(time.time()-t1)



