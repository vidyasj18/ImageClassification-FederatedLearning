import tensorflow as tf

print("Loading global FL model...")
model = tf.keras.models.load_model("global_model/model.h5")
print("Model loaded!")

# Load CIFAR-10 test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

# Evaluate global model
loss, acc = model.evaluate(x_test, y_test, verbose=2)

print("\n FEDERATED MODEL RESULTS ")
print("Test Loss:", loss)
print("Test Accuracy:", acc)
