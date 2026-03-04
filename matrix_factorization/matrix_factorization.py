import time
import numpy as np
import scipy
import matplotlib.pyplot as plt

train = scipy.sparse.load_npz('../processed_20k_4k/train_user_movie.npz')
train_csr = train.tocsr()
train_csc = train.tocsc()
test = scipy.sparse.load_npz('../processed_20k_4k/test_user_movie.npz')

K = 25
l2_lambda = 0.01
epochs = 20
N, M = train.shape
rng = np.random.default_rng(48)
W = rng.random((N, K))
U = rng.random((M, K))
b = rng.random(N)
c = rng.random(M)
mu = np.mean(train.data)

train_loss = []
test_loss = []

t1 = time.time()

for epoch in range(epochs):
    print('Epoch', epoch)

    for i in range(N):
        row = train_csr[i, :]
        idx = row.coords[0]
        u_j = U[idx]
        n_mov = len(idx)

        A = u_j.transpose() @ u_j + l2_lambda*np.eye(K)

        resid = row.data - b[i] - c[idx] - mu
        vec = resid @ u_j

        # update W
        W[i] = np.linalg.solve(A, vec)
        # update b
        b[i] = np.sum(resid + b[i] - W[i] @ u_j.transpose())/(n_mov + l2_lambda)

    print('Updated W, b')
    print(time.time()-t1)

    for j in range(M):
        col = train_csc[:, j]
        idx = col.coords[0]
        w_i = W[idx]
        n_use = len(idx)

        A = w_i.transpose() @ w_i + l2_lambda*np.eye(K)

        resid = col.data - b[idx] - c[j] - mu
        vec = resid.transpose() @ w_i

        # update U
        U[j] = np.linalg.solve(A, vec)
        # update c
        c[j] = np.sum(resid + c[j] - w_i @ U[j].transpose())/(n_use + l2_lambda)

    print('Updated U, c')
    print(time.time()-t1)

    # get training loss
    rows, cols = train.coords
    vals = train.data
    preds = mu + b[rows] + c[cols] + np.einsum('ij,ij->i', W[rows], U[cols])
    mse = np.mean((vals - preds)**2)
    train_loss.append(mse)

    print('Train loss')
    print(time.time()-t1)

    # get testing loss
    rows, cols = test.coords
    vals = test.data
    preds = mu + b[rows] + c[cols] + np.einsum('ij,ij->i', W[rows], U[cols])
    mse = np.mean((vals - preds)**2)
    test_loss.append(mse)

    print('Test Loss')
    print(time.time()-t1)

plt.plot(range(epochs), train_loss, label = 'training')
plt.plot(range(epochs), test_loss, label = 'testing')
plt.legend()
print('Final Training Loss:', train_loss[-1])
print('Final Testing Loss:', test_loss[-1])

np.save('MF_W', W)
np.save('MF_U', U)
np.save('MF_b', b)
np.save('MF_c', c)
np.save('MF_mu', mu)

plt.title('ALS Mean Squared Error')
plt.savefig('ALS_MSE.png')
plt.show()
