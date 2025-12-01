import flwr as fl
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# defining the function

def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:10000] / 255.0, y_train[:10000]  # smaller dataset
x_test, y_test = x_test[:2000] / 255.0, y_test[:2000]


# flower client class
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = model_fn()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": acc}


# start client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=CifarClient()
)
