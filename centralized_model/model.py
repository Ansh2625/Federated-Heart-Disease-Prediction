import tensorflow as tf
from keras import layers, Model

class HeartDiseaseModel(Model):
    def __init__(self, input_dim):
        super(HeartDiseaseModel, self).__init__()
        self.norm = layers.BatchNormalization()

        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)

        self.dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)

        self.residual_dense = layers.Dense(64, activation='relu')  # same size for residual

        self.concat_layer = layers.Concatenate()

        self.dense3 = layers.Dense(32, activation='relu')
        self.dropout3 = layers.Dropout(0.2)

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.norm(inputs, training=training)

        x1 = self.dense1(x)
        x1 = self.dropout1(x1, training=training)

        x2 = self.dense2(x1)
        x2 = self.dropout2(x2, training=training)

        # Residual connection
        x2 = x2 + self.residual_dense(x1)

        x3 = self.dense3(x2)
        x3 = self.dropout3(x3, training=training)

        concat = self.concat_layer([x1, x2, x3])  # adds dimensionality

        output = self.output_layer(concat)
        return output
