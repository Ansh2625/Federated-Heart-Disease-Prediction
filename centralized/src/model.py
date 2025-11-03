import tensorflow as tf
from tensorflow.keras import layers, Model

class GLUBlock(layers.Layer):
    """Gated Linear Unit block used in TabNet"""
    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.fc = layers.Dense(units * 2, activation=None)
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x = self.fc(x)
        x = self.bn(x, training=training)
        x = self.drop(x, training=training)
        out, gate = tf.split(x, 2, axis=-1)
        return out * tf.nn.sigmoid(gate)


class FeatureTransformer(layers.Layer):
    """Stack of GLU blocks"""
    def __init__(self, units, n_glu=2, dropout_rate=0.1):
        super().__init__()
        self.blocks = [GLUBlock(units, dropout_rate) for _ in range(n_glu)]

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x


class TabNet(Model):
    def __init__(self, input_dim, feature_dim=64, output_dim=32, n_steps=5, relaxation_factor=1.5):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.relax = relaxation_factor

        # Shared feature transformer
        self.shared_ft = FeatureTransformer(feature_dim, n_glu=2)

        # Step-specific transformers
        self.step_ft = [FeatureTransformer(feature_dim, n_glu=2) for _ in range(n_steps)]
        self.att_transform = [layers.Dense(input_dim, activation="softmax") for _ in range(n_steps)]

        # Final classifier
        self.fc = layers.Dense(1, activation="sigmoid")

    def call(self, x, training=False):
        prior = tf.zeros_like(x)
        out_agg = 0
        masked_x = x

        for step in range(self.n_steps):
            h = self.shared_ft(masked_x, training=training)
            h = self.step_ft[step](h, training=training)

            out = tf.keras.activations.relu(h[:, :self.output_dim])
            out_agg += out

            mask = self.att_transform[step](h)
            mask = mask * (prior + 1e-6)
            mask = mask / tf.reduce_sum(mask, axis=-1, keepdims=True)
            prior += (1 - mask) * self.relax

            masked_x = x * mask

        return self.fc(out_agg)


# Wrapper so train.py still works
class HeartDiseaseModel(TabNet):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim)
