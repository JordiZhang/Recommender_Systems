from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Activation, Dot, Add, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import scipy
import matplotlib.pyplot as plt
import numpy as np

train = scipy.sparse.load_npz('../processed_20k_4k/train_user_movie.npz')
test = scipy.sparse.load_npz('../processed_20k_4k/test_user_movie.npz')

N, M = train.shape
K = 10
l2_lambda = 1e-6
mu = np.mean(train.data)
epochs = 25

# inputs
u = Input(shape = (1,))
m = Input(shape = (1,))

# user movie matrices
W = Embedding(N, K, embeddings_regularizer=l2(l2_lambda))(u)
U = Embedding(M, K, embeddings_regularizer=l2(l2_lambda))(m)

# biases
b = Embedding(N, 1, embeddings_regularizer=l2(l2_lambda))(u)
c = Embedding(M, 1, embeddings_regularizer=l2(l2_lambda))(m)

# matrix factorization
x = Dot(axes = 2)([W, U])
x = Add()([x, b, c])
x = Flatten()(x)

# neural network
emb_u = Flatten()(W)
emb_m = Flatten()(U)
y = Concatenate()([emb_u, emb_m])
y = Dense(400)(y)
y = Dropout(0.25)(y)
y = Dense(100)(y)
y = Activation('relu')(y)
y = Dense(1)(y)

# combine into neuMF
x = Concatenate()([x, y])
x = Dense(25)(x)
x = Activation('relu')(x)
x = Dense(1)(x)

# build model
model = Model(inputs = [u, m], outputs = x)
model.compile(
    loss = 'mse',
    optimizer = Adam(),
    metrics = ['mse']
)



r = model.fit(
    x = train.coords,
    y = train.data,
    epochs = epochs,
    batch_size = 128,
    validation_data = (test.coords, test.data)
)

plt.plot(r.history['loss'], label = 'Training loss')
plt.plot(r.history['val_loss'], label = 'Testing loss')
plt.legend()
plt.show()

plt.plot(r.history['mse'], label = 'Training MSE')
plt.plot(r.history['val_mse'], label = 'Testing MSE')
plt.title('Neural CF Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('Neural_CF_MSE.png')
plt.show()



