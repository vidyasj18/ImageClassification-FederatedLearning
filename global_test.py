import tensorflow as tf
from model import create_model

model = create_model()
model.load_weights("global_model/model.h5")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

loss, acc = model.evaluate(x_test, y_test, verbose=2)

print("Global model test accuracy:", acc)
