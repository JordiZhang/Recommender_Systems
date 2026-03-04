from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import scipy
import numpy as np
import matplotlib.pyplot as plt

train = scipy.sparse.load_npz('../processed_20k_4k/train_user_movie.npz')
test = scipy.sparse.load_npz('../processed_20k_4k/test_user_movie.npz')

N, M = train.shape
K = 25
# regularization is applied to entire embedding matrix, so if we want to use regularization, lambda has to be incredibly small, like 1e-6
l2_lambda = 1e-6
mu = np.mean(train.data)
epochs = 10

# inputs
u = Input(shape = (1,))
m = Input(shape = (1,))

# user movie matrices
W = Embedding(N, K, embeddings_regularizer=l2(l2_lambda))(u)
U = Embedding(M, K, embeddings_regularizer=l2(l2_lambda))(m)

# biases
b = Embedding(N, 1, embeddings_regularizer=l2(l2_lambda))(u)
c = Embedding(M, 1, embeddings_regularizer=l2(l2_lambda))(m)

# prediction
x = Dot(axes = 2)([W, U])
x = Add()([x, b, c])
x = Flatten()(x)

# build model
model = Model(inputs = [u, m], outputs = x)
model.compile(
    loss = 'mse',
    optimizer = Adam(),
    metrics = ['mse']
)

y_train = (train.data - mu).reshape(-1, 1)
y_test = (test.data - mu).reshape(-1, 1)

r = model.fit(
    x = train.coords,
    y = y_train,
    epochs = epochs,
    batch_size = 128,
    validation_data = (test.coords, y_test)
)

plt.plot(r.history['loss'], label = 'Training loss')
plt.plot(r.history['val_loss'], label = 'Testing loss')
plt.legend()
plt.show()

plt.plot(r.history['mse'], label = 'Training MSE')
plt.plot(r.history['val_mse'], label = 'Testing MSE')
plt.title('MF Keras Mean Squared Error')
plt.legend()
plt.savefig('Keras_MSE.png')
plt.show()



