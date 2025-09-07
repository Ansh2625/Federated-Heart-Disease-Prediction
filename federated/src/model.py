import tensorflow as tf
from keras import layers, Model, regularizers

class DenseBlock(layers.Layer):
    def __init__(self, units, dropout, l2=1e-4):
        super().__init__()
        self.bn = layers.BatchNormalization()
        self.dense = layers.Dense(
            units, activation="relu", kernel_regularizer=regularizers.l2(l2)
        )
        self.do = layers.Dropout(dropout)

    def call(self, x, training=False):
        z = self.bn(x, training=training)
        z = self.dense(z)
        z = self.do(z, training=training)
        return z

class HeartDiseaseModel(Model):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_norm = layers.BatchNormalization()

        # Block 1
        self.b1 = DenseBlock(256, dropout=0.40, l2=1e-4)

        # Block 2 + residual from block1
        self.b2_dense = layers.Dense(
            128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)
        )
        self.b2_bn = layers.BatchNormalization()
        self.b2_do = layers.Dropout(0.30)
        self.res_proj = layers.Dense(128, activation=None)

        # Block 3
        self.b3 = DenseBlock(64, dropout=0.20, l2=1e-4)

        # Output
        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.input_norm(inputs, training=training)
        h1 = self.b1(x, training=training)
        h2 = self.b2_dense(h1)
        h2 = self.b2_bn(h2, training=training)
        h2 = tf.nn.relu(h2)
        h2 = self.b2_do(h2, training=training)
        h2 = h2 + self.res_proj(h1)  # residual
        h3 = self.b3(h2, training=training)
        return self.out(h3)
