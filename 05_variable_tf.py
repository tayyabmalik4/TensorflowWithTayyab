# (05)*****************************Variable in tensorflow*********************

# ----Definition----The variable() constructor requires an initial value for the variable,which can be a tensor of any type nd shape. This initial value defines the tyoe and shape of the variable. After construction, the type and shape of the variable are fixed.The value can be changed using one of the assign methods.


# ----importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape

# ----Creating tensor
tf_var=tf.Variable(34)
# print(tf_var)

# ----Creating 1d array
tf_1d=tf.Variable([1,2,3,4])
# print(tf_1d)

# ----to checking the name and shape and dtype of the variable 
# print(tf_1d.name)
# print(tf_1d.shape)
# print(tf_1d.dtype)
# ----Converting tensorflow arrray to numpy array
# print(tf_1d.numpy())
# print(type(tf_1d))

# ----Creating TF using Integer,Float,String,Bool,Complex Number
tf_int=tf.Variable([1,2,3,4,5])
# print(tf_int)
tf_float=tf.Variable([1.3,1.4,45,2.6])
# print(tf_float)
tf_str=tf.Variable(['a','b','c','d'])
# print(tf_str)
tf_bool=tf.Variable([True,False])
# print(tf_bool)
# ----we use the j in the place of Complec number(i(ioata))
tf_cx=tf.Variable([3+4j])
# print(tf_cx)


# -----Using Constant variable to converd to Tensorflow variable 
tf_const=tf.constant([1,2,3,4])
# print(tf_const)
# print(tf.Variable(tf_const))

# -----Creat Tensorflow variable with different shape
tf_2d=tf.Variable([[2,3],[3,4]])
# print(tf_2d)

# -----Now we creat a shape as we wish and then change also dtype
tf_2dd=tf.Variable([[2,3],[4,5]],shape=(2,2), dtype='float64')
# print(tf_2dd)


# ----Converting 1d to 2d array
Sultan= tf.Variable([786, 147,3.22,45])
# print(tf.reshape(Sultan, (2,2)))

# ---checking higist value index than we use argmax
max_value=tf.argmax(tf_2dd)
# print(max_value)

# ----Viewed and Convert as a tensor
convrt=tf.convert_to_tensor(tf_2dd)
# print(convrt)

# ----change and assign new value to tensor
# ----we can't change shape and type of the tensorflow
# ----when we use assigned the value than the value is replacing the previous value
assign=tf_1d.assign([2,4,6,8])
# print(assign)
# print(tf_1d)

# ----when we want to chnage variable also add the numbers than we use assign
# ----if we want to subtract the numbers than we use assign_sub function
assign_add=tf_1d.assign_add([2,5,3,4])
# print(assign_add)
# print(tf_1d)



# -------------------END--------------------------------