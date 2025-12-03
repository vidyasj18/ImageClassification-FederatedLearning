import tensorflow as tf
from model import create_model

# used to varify global performance on the test data

model = create_model()
model.load_weights("global_model/model.h5")

# load cifar test datset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

# evaluate the global model

loss, acc = model.evaluate(x_test, y_test, verbose=2)
print("Global model test accuracy:", acc)
