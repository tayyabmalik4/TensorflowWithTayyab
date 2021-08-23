# (08)*********************Ones and ones-like and ones-initializer in tensorflow*******************

# -----if we want to creat tensor which all elements should be ones then we use ones function

#  ----Creat Tensor with every elements are one in Tensorflow 2.X
# ----Syntax: tf.ones()

# -----importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops_v2 import Initializer

# ----1d tensor
one=tf.ones((1),tf.int32)
# print(one)
# ----2d tensor
one2=tf.ones((2,2),tf.float32, )
# print(one2)

# ----3d tensor
one3=tf.ones((2,2,4),tf.float32)
# print(one3)

# ----if we don't know the shape of ones tensor we use ones_like function in the constant tensor
con_2dt=tf.constant([[1,2,3],[4,5,6]])
# print(con_2dt)
one_lik=tf.ones_like(con_2dt,tf.float32)
# print(one_lik)
var_2dt=tf.Variable([[1,2,3],[4,5,6]])
# print(var_2dt)
one_lik2=tf.ones_like(var_2dt,tf.float32)
# print(one_lik2)


# -----Ones_initializer()
# -----Initializer that generates tensors initialized to 1.
# -----Initializers allow you to pre-specify an initialization strategy, encoded in the Initializer object, without knowing the shape and dtype of the variable being initialized.
# ----first of all we creat a function and define it 
def ones_variable(shape,dtype,initializer):
    return tf.Variable(initializer(shape=shape,dtype=dtype))
one_init=ones_variable((2,2),tf.float32,tf.ones_initializer())
# print(one_init)
# ----Creating 3d variable in ones_initializer
one_init3=ones_variable((2,3,3),tf.float32,tf.ones_initializer())
# print(one_init3)

# ----if we want to use ones_initializer in one line then we use this method
one=tf.Variable(tf.ones_initializer()((3,4),tf.float64))
print(one)





# ------------------------------END---------------------------------