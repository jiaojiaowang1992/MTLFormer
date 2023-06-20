
import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)#多头注意力
        # 前馈神经网络
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        #残差网络
        self.layernorm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_b = layers.LayerNormalization(epsilon=1e-6)
        #丢失部分单元
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layernorm_b(out_a + ffn_output)

class TokenAndPositionEmbedding(layers.Layer): #位置编码
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        #编码层
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #位置编码层
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_predict_model(max_case_length, vocab_size, output_dim,
    embed_dim = 36, num_heads = 4, ff_dim = 64):
    #输入
    inputs1 = layers.Input(shape=(max_case_length,))
    inputs2 = layers.Input(shape=(max_case_length,))
    time_inputs = layers.Input(shape=(3,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs1) #编码层
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x) #Transformer层
    x = layers.GlobalAveragePooling1D()(x) #全局平均池化
    x1 = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs2)
    x1 = TransformerBlock(embed_dim, num_heads, ff_dim)(x1)
    x1 = layers.GlobalAveragePooling1D()(x1)

    #next_act特征
    x_1 = layers.Dropout(0.1)(x)
    x_1 = layers.Dense(32, activation="relu", name="next_act", kernel_regularizer='l2')(x_1)

    #next_time特征
    x_t1 = layers.Dense(32, activation="relu")(time_inputs)
    x_2 = layers.Concatenate()([x1, x_t1])
    x_2 = layers.Dropout(0.1)(x_2)
    x_2 = layers.Dense(64, activation="relu", name="next_time", kernel_regularizer='l2')(x_2)

    #remain_time特征
    x_t2 = layers.Dense(32, activation="relu")(time_inputs)
    x_3 = layers.Concatenate()([x1, x_t2])
    x_3 = layers.Dropout(0.1)(x_3)
    x_3 = layers.Dense(64, activation="relu", name="remain_time", kernel_regularizer='l2')(x_3)

    #共享
    x = layers.concatenate([x_1, x_2, x_3])


    #输出
    out1 = layers.Dropout(0.1)(x)
    out1 = layers.Dense(128, activation="relu")(out1)
    out1 = layers.Dropout(0.1)(out1)
    out1 = layers.Dense(32, activation="relu")(out1)
    out1 = layers.Dropout(0.1)(out1)
    outputs1 = layers.Dense(output_dim, activation="linear", name = 'out1')(out1)

    out2 = layers.Dropout(0.1)(x)
    out2 = layers.Dense(128, activation="relu")(out2)
    out2 = layers.Dropout(0.1)(out2)
    out2 = layers.Dense(32, activation="relu")(out2)
    out2 = layers.Dropout(0.1)(out2)
    outputs2 = layers.Dense(1, activation="linear", name="out2")(out2)

    out3 = layers.Dropout(0.1)(x)
    out3 = layers.Dense(128, activation="relu")(out3)
    out3 = layers.Dropout(0.1)(out3)
    out3 = layers.Dense(32, activation="relu")(out3)
    out3 = layers.Dropout(0.1)(out3)
    outputs3 = layers.Dense(1, activation="linear", name="out3")(out3)

    transformer = tf.keras.Model(inputs=[inputs1, inputs2, time_inputs], outputs=[outputs1,outputs2,outputs3],
        name = "multitask_transformer")
    return transformer

