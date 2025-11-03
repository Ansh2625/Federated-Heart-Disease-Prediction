# centralized/src/model_2.py
import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads=4, dropout=0.3, ffn_mult=4):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
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
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

class SAINT(Model):
    """Medium-heavy SAINT for tabular data (8 layers, 256 dim)."""
    def __init__(self, input_dim, depth=8, dim=256, heads=4, dropout=0.3, ffn_mult=4):
        super().__init__()
        self.input_proj = layers.Dense(dim)

        self.transformers = [
            TransformerBlock(dim=dim, heads=heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(depth)
        ]

        self.global_pool = layers.GlobalAveragePooling1D()
        self.norm = layers.LayerNormalization()
        self.out = layers.Dense(1, activation="sigmoid")

    def call(self, x, training=False):
        # expand to sequence length 1
        x = tf.expand_dims(x, axis=1)
        h = self.input_proj(x)
        for blk in self.transformers:
            h = blk(h, training=training)
        h = self.global_pool(h)
        h = self.norm(h)
        return self.out(h)
