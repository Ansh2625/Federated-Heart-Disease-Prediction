# federated/src/model_2.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads=4, dropout=0.3, ffn_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.attn  = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
        self.norm1 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dim * ffn_mult, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])
        self.norm2 = layers.LayerNormalization()
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.norm1(x + self.drop1(self.attn(x, x, training=training), training=training))
        x = self.norm2(x + self.drop2(self.ffn(x, training=training), training=training))
        return x

class SAINT(Model):
    def __init__(self, input_dim, depth=8, dim=256, heads=4, dropout=0.3, ffn_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = layers.Dense(dim)
        self.blocks = [TransformerBlock(dim=dim, heads=heads, dropout=dropout, ffn_mult=ffn_mult)
                       for _ in range(depth)]
        self.pool = layers.GlobalAveragePooling1D()
        self.norm = layers.LayerNormalization()
        self.out  = layers.Dense(1, activation="sigmoid")

    def call(self, x, training=False):
        x = tf.expand_dims(x, axis=1)  # (B,1,F)
        h = self.input_proj(x)
        for b in self.blocks:
            h = b(h, training=training)
        h = self.pool(h)
        h = self.norm(h)
        return self.out(h)
