import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

class HeartDiseaseModel(Model):

    def __init__(self):
        super(HeartDiseaseModel, self).__init__()

        self.input_layer = Dense(128, activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.4)

        self.hidden1 = Dense(64, activation='relu')
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)

        self.hidden2 = Dense(32, activation='relu')
        self.bn3 = BatchNormalization()
        self.dropout3 = Dropout(0.2)

        self.hidden3 = Dense(16, activation='relu')

        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.hidden1(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.hidden2(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)

        x = self.hidden3(x)

        return self.output_layer(x)