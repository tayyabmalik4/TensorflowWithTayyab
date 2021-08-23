# (09)*************************Zeros and Zeros_like and Zeros_initializer in Tensorflow**************

# -----Creat Tensor with every elements are zeros in TensorFlow 2.X
# -----Syntax: tf.zeros(shape,dtype)


# -----inporting libraires
import numpy as np
import tensorflow as tf

# ----Creat a tensor with all elements set to zero
# ----Creat 1D zeros
zero1=tf.zeros((2),dtype=tf.int32)
# print(zero1)
# ----Creating 2D zeros
zero2=tf.zeros((2,2),dtype=tf.float32)
# print(zero2)

# ----Creating 3D zeros
zero3=tf.zeros((2,3,3),dtype=tf.float64)
# print(zero3)

# ----If we want to check the shape of the variable then we use shape function
# print(zero3.shape)
# ----if we want to check the dtype of the variable then we use dtype function
# print(zero3.dtype)
# ----if we want to check out just the numpy values then we use numpy function
# print(zero3.numpy())
# ----if we want to check out the numpy array in the form of list then we use to_list function
# print(zero2.numpy().tolist())

# ----Creating zeros_like tensor using constant and variable tensors
# ----Creating constant
con=tf.constant([[1,2,3,4],[4,5,6,7],[6,4,7,8]],dtype=tf.float32)
# print(con)
# ----Now Creating zeros_like tensor
zero_like=tf.zeros_like(con)
# print(zero_like)
# ----Creating variable tensor
var=tf.Variable([[1,2,3],[3,4,5],[5,6,7]],dtype=tf.int32)
# print(var)
zero_like2=tf.zeros_like(var)
# print(zero_like2)


# ----Creating zeros_initialinzer
# ----tf.zeros_initializer
# ----But we did not put values dirextly so we creat a function and then other parameters put as a input
# ----Initializer allow you to pre-specify an initialization strategy, encoded in the Initialer object, without knowing the shape and dtype of the variable being initialized
def zeros_variable(shape,dtype,initializer):
    return tf.Variable(initializer(shape=shape,dtype=dtype))
zero=zeros_variable((2,3,3),tf.int32,tf.zeros_initializer())
# print(zero)

# -----If we want to creat a zeros_initializer than we use this menthod
zero0=tf.Variable(tf.zeros_initializer()((2,2),tf.float64))
print(zero0)
