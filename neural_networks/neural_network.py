from keras.models import Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Activation
from keras.optimizers import Adam, SGD
import scipy
import matplotlib.pyplot as plt

train = scipy.sparse.load_npz('../processed_20k_4k/train_user_movie.npz')
test = scipy.sparse.load_npz('../processed_20k_4k/test_user_movie.npz')

N, M = train.shape
K = 10
epochs = 25

# inputs
u = Input(shape = (1,))
m = Input(shape = (1,))

# user movie matrices
emb_u = Embedding(N, K)(u)
emb_m = Embedding(M, K)(m)
emb_u = Flatten()(emb_u)
emb_m = Flatten()(emb_m)

x = Concatenate()([emb_u, emb_m])
x = Dense(400)(x)
x = Dense(100)(x)
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
plt.title('Neural Network Mean Squared Error')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('NN_MSE.png')
plt.show()



