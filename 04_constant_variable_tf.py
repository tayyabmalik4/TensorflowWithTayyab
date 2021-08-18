# (04)****************************Constant variable in Tensorflow************************

# -----Definition----Creates a constant tensor from a tensor-like object
# -----Syntax:----tf.constant()

# -----How to create constant variable in Tensorflow 2.x
# -----tf.constant():
# -----Agrs
# ----value(A constant value(or list) of output type dtype),
# -----dtype(The type of the elements of the resulting tensor),
# -----shape(Optinal dimentinals of resulting tensor),
# -----name(Optional name forn the tensor)

# -----Returns:---A Constant Tensor.
# -----Link-----https://www.tensorflow.org/api_docs/python/tf/constant

# ----importing library
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf

# ----checking version
version=tf.__version__
# -----Current version is 2.6.0
# print(version)    

# ------------for checking the gpu is awailable or not
# print(tf.test.is_gpu_available())

# -----Creating Integer Constant
integ=tf.constant(10)
# print(integ)

# ----Creating float constant
flot=tf.constant(10.5)
# print(flot)

# ----creating string constant
strn=tf.constant("Hey! My company name is Sultan's Production")
# print(strn)

# ----Creating Bool constant
bol=tf.constant(True)
# print(bol)

# ----Creat constant Numpy array/list/tuple
np_array=tf.constant(np.array([1,2.5,3,4,5]))
# print(np_array)

# ----creating ID array in tensorflow
t_1d=tf.constant([1,2,3,4])
# print(t_1d)

# ----now creating 2d array in tensorflow
t_2d=tf.constant([[1,2],[3,4]])
# print(t_2d)

# ----Now we converd 1d arrray to 2d array in tensorflow
t_2d_1=tf.constant([[1,2,3,4,5,6,7,8]],shape=(4,2))
# print(t_2d_1)

# ----Now we define datatype in the tensorflow constant
t_2d_11=tf.constant([1,2,3,4],shape=(2,2),dtype='float64')
# print(t_2d_11)

# ----Now we creating N_dimentional array in tensorflow
t_3d_1=tf.constant([[[1,2],[2,3],[3,4]]])
# print(t_3d_1)

# ----Now we check that the type of variable in tensorflow
# print(type(t_3d_1))

# ----Now we check the shape of constant
# print(t_3d_1.shape)

# ----If we want to printout just exct values then we use numpy library
# print(t_2d_1.numpy())

# ----If we want to acces the spacific value of the array then we use this method
# print(t_2d_1.numpy()[2])

# ----Now we find out the datatype then we use dtype function
print(t_2d_1.dtype )

