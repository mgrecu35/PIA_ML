import tf.keras
import tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[9, 9, 10]),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(filters=64,kernel_size=3),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
])
