import keras.src.layers
import tensorflow as tf
import keras as ks
from keras import Input
from keras.src.layers import Dense, MaxPooling2D, Conv3D
from tensorflow.python.types.core import Tensor




def main(k: [Tensor], result:[Tensor], input_tensor=None) -> None:
 convolution2a = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(input_tensor)
 max_pooling = MaxPooling2D(pool_size=(2, 2))(convolution2a)
 shape = (2, 16, 8, 4) # Creating a tensor, first variable represents the time-step, second variable represents the speech, third variable represents a peek operation, fourth is the kernel obv!!
 shape = Input(shape=shape)
 convolution2a.input(shape)
 convolution2a.add_loss(lambda x,y: tf.sigmoid(tf.matmul(x, tf.reduce_mean(x, axis=[1,2,3], keepsdims=True, name="meanops"))))
 output = convolution2a.output()
 max_pooling.input(output)
 output = max_pooling.output()
 variable2 = tf.Variable(tf.sigmoid(output), True, True)
 variable2 = tf.Variable(tf.tensor(variable2), True, True)
 for i in range(len(k)):
 vectorarr = [k[i], k[i + 1]]
 i = tf.matmul(variable2, keras.src.ops.concatenate(vectorarr, -1))
 i = i, tf.boolean_mask(i, [False, True], 1, "sigmoid_mask")
 return k
def model() -> keras.Model:
 query = Input(shape=(10, 10, 10))
 value = Input(shape=(10, 30, 60))
 key = Input(shape=(5, 6, 20))

 a = Dense(64)
 a.add_variable(shape=tf.sigmoid(tf.eye), initializer=tf.keras.initializers.Identity, dtype=None, trainable=True, name="activationWeight")
 a.add_loss(lambda x: tf.sigmoid(x))
 m = keras.Model(
 Dense(64),
 keras.layers.Attention(
 use_scale=False, score_mode="dot", dropout=0.0, seed=None
 ),
 keras.layers.GroupQueryAttention(
 head_dim=60, num_query_heads=10, num_key_value_heads=15, dropout=0.0, use_bias=True, flash_attention=None,
 kernel_initializer="glorot-uniform", bias_initializer="zeroes", kernel_regularizer=None, bias_regularizer=None,
 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seed=1, use_casual_mask=True
 ),
 Dense(64)
 .add_variable(shape=tf.sigmoid(tf.eye), initializer=tf.keras.initializers.Identity, dtype=None, trainable=True, name="activationWeight")
 .add_loss(lambda x: tf.sigmoid(x))
 )
 return m
def compile(model: keras.Model) -> None:
 model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 model.summary()
