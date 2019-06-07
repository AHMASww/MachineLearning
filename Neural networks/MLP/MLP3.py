import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])

# 基础模型
model = keras.Sequential([
    layers.Dense(64, activation="sigmoid", kernel_initializer="he_normal", input_shape=(784,)),
    layers.Dense(64, activation="sigmoid", kernel_initializer="he_normal"),
    layers.Dense(64, activation="sigmoid", kernel_initializer="he_normal"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=keras.optimizers.radma(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, batch_size = 256, epochs=200, validation_split=0.3, verbose=1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "validation"], loc="upper left")
plt.show()

result = model.evaluate(x_test, y_test)