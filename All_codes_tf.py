import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf




# (04)****************************Constant variable in Tensorflow************************

version=tf.__version__
print(version)
print(tf.test.is_gpu_available())
integ=tf.constant(10)
print(integ)
flot=tf.constant(10.5)
print(flot)
strn=tf.constant("Hey! My company name is Sultan's Production")
print(strn)
bol=tf.constant(True)
print(bol)
np_array=tf.constant(np.array([1,2.5,3,4,5]))
print(np_array)
t_1d=tf.constant([1,2,3,4])
print(t_1d)
t_2d=tf.constant([[1,2],[3,4]])
print(t_2d)
t_2d_1=tf.constant([[1,2,3,4,5,6,7,8]],shape=(4,2))
print(t_2d_1)
t_2d_11=tf.constant([1,2,3,4],shape=(2,2),dtype='float64')
print(t_2d_11)
t_3d_1=tf.constant([[[1,2],[2,3],[3,4]]])
print(t_3d_1)
print(type(t_3d_1))
print(t_3d_1.shape)
print(t_2d_1.numpy())
print(t_2d_1.numpy()[2])
print(t_2d_1.dtype )


# (05)*****************************Variable in tensorflow*********************


tf_var=tf.Variable(34)
print(tf_var)
tf_1d=tf.Variable([1,2,3,4])
print(tf_1d)
print(tf_1d.name)
print(tf_1d.shape)
print(tf_1d.dtype)
print(tf_1d.numpy())
print(type(tf_1d))
tf_int=tf.Variable([1,2,3,4,5])
print(tf_int)
tf_float=tf.Variable([1.3,1.4,45,2.6])
print(tf_float)
tf_str=tf.Variable(['a','b','c','d'])
print(tf_str)
tf_bool=tf.Variable([True,False])
print(tf_bool)
tf_cx=tf.Variable([3+4j])
print(tf_cx)
tf_const=tf.constant([1,2,3,4])
print(tf_const)
print(tf.Variable(tf_const))
tf_2d=tf.Variable([[2,3],[3,4]])
print(tf_2d)
tf_2dd=tf.Variable([[2,3],[4,5]],shape=(2,2), dtype='float64')
print(tf_2dd)
Sultan= tf.Variable([786, 147,3.22,45])
print(tf.reshape(Sultan, (2,2)))
max_value=tf.argmax(tf_2dd)
print(max_value)
convrt=tf.convert_to_tensor(tf_2dd)
print(convrt)
assign=tf_1d.assign([2,4,6,8])
print(assign)
print(tf_1d)
assign_add=tf_1d.assign_add([2,5,3,4])
print(assign_add)
print(tf_1d)


# (06)********************Creat placeholder in tensorflow*************************
# -----Syntex:-----tf.compat.v1.placeholder(dtype,shape=None, name=None)


tf.compat.v1.disable_eager_execution()
a=tf.compat.v1.placeholder(dtype=tf.float32, shape=(400,400))
print(a)
print(a.dtype)
print(a.shape)
b=tf.compat.v1.placeholder(dtype=tf.float32, shape=(400,400))
print(b)
c=tf.add(a,b)
print(c)
ones_array=np.ones((400,400),np.float32)
print(ones_array)
with tf.compat.v1.Session() as sess:
    d= sess.run(c, feed_dict={a:ones_array, b:ones_array})
print(d)


# (07)*******************Sparce Tensor in Tensorflow*****************************

st1=tf.SparseTensor(indices=[[0,3],[2,4]],values=[10,20],dense_shape=[3,10])
print(st1)
np_array=np.array([[1,4,3,2],[2,1,3,7],[9,8,1,2],[4,9,2,1]])
print(np_array)
st2_fd=tf.sparse.from_dense(np_array)
print(st2_fd)
print(st2_fd.values)
print(st2_fd.values.numpy())
print(st2_fd.values.numpy().tolist())
print(st2_fd.indices)
print(st2_fd.indices.numpy())
print(st2_fd.indices.numpy().tolist())
print(st2_fd.dense_shape)
print(st2_fd.dense_shape.numpy())
print(st2_fd.dense_shape.numpy().tolist())
dt_fst=tf.sparse.to_dense(st2_fd)
print(dt_fst)
print(dt_fst.numpy())
print(dt_fst.shape)
print(dt_fst.dtype)
ad=tf.sparse.add(st2_fd,st2_fd)
print(ad)
d=tf.sparse.to_dense(ad)
print(d)
print(d.numpy())


# (08)*********************Ones and ones-like and ones-initializer in tensorflow*******************

one=tf.ones((1),tf.int32)
print(one)
one2=tf.ones((2,2),tf.float32, )
print(one2)
one3=tf.ones((2,2,4),tf.float32)
print(one3)
con_2dt=tf.constant([[1,2,3],[4,5,6]])
print(con_2dt)
one_lik=tf.ones_like(con_2dt,tf.float32)
print(one_lik)
var_2dt=tf.Variable([[1,2,3],[4,5,6]])
print(var_2dt)
one_lik2=tf.ones_like(var_2dt,tf.float32)
print(one_lik2)
def ones_variable(shape,dtype,initializer):
    return tf.Variable(initializer(shape=shape,dtype=dtype))
one_init=ones_variable((2,2),tf.float32,tf.ones_initializer())
print(one_init)
one_init3=ones_variable((2,3,3),tf.float32,tf.ones_initializer())
print(one_init3)
one=tf.Variable(tf.ones_initializer()((3,4),tf.float64))
print(one)


# (09)*************************Zeros and Zeros_like and Zeros_initializer in Tensorflow**************

zero1=tf.zeros((2),dtype=tf.int32)
print(zero1)
zero2=tf.zeros((2,2),dtype=tf.float32)
print(zero2)
zero3=tf.zeros((2,3,3),dtype=tf.float64)
print(zero3)
print(zero3.shape)
print(zero3.dtype)
print(zero3.numpy())
print(zero2.numpy().tolist())
con=tf.constant([[1,2,3,4],[4,5,6,7],[6,4,7,8]],dtype=tf.float32)
print(con)
zero_like=tf.zeros_like(con)
print(zero_like)
var=tf.Variable([[1,2,3],[3,4,5],[5,6,7]],dtype=tf.int32)
print(var)
zero_like2=tf.zeros_like(var)
print(zero_like2)
def zeros_variable(shape,dtype,initializer):
    return tf.Variable(initializer(shape=shape,dtype=dtype))
zero=zeros_variable((2,3,3),tf.int32,tf.zeros_initializer())
print(zero)
zero0=tf.Variable(tf.zeros_initializer()((2,2),tf.float64))
print(zero0)






# -------------END----------------------