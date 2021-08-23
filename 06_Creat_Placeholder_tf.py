# (06)********************Creat placeholder in tensorflow*************************

# ----placeholder is not available in tensorflow 2.X version. but it is available in 1.X version
# ----We use placeholder in 2.X as the 1.X
# -----Definition-----A placeholder is simply a variable that we will assign data to at a later date.It allows us to creat out operations and build our computation graph, without needing the data.

# ----internet Definition----Intuitively, a placeholder is a pre-formatted container into which content can be placed. By providing pre-set formatting to its content, it places many of the formatting choices in the hands of the template designer while allowing the end-user to concentrate on the actual content.

# -----Syntex:-----tf.compat.v1.placeholder(dtype,shape=None, name=None)

# ----importing libraries
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


# ----Placeholders are not executable immediately so we need to disable eager exicution in TF 2 not in 1
# ----as we know that the placeholder is not run in the tensorflow 2.X so error dosnot arrure we use eager exicution
tf.compat.v1.disable_eager_execution()

# ----Creating Placeholder
a=tf.compat.v1.placeholder(dtype=tf.float32, shape=(400,400))
# print(a)
# ----if we want to check the dtype than we use dtype[] function
# print(a.dtype)
# ----if we want to check the shape of the tensorflow we use shape function
# print(a.shape)
# ----if we want to check the name of the tensorflow we use name function
b=tf.compat.v1.placeholder(dtype=tf.float32, shape=(400,400))
# print(b)

# -----Now perform mathematical operation with placeholder
c=tf.add(a,b)
# print(c)

# -----Creat Numpy array
ones_array=np.ones((400,400),np.float32)
# print(ones_array)

# -----Exicute Tensorflow placeholder using session
# -----we use session because we use the 1.X version of Tensorflow
with tf.compat.v1.Session() as sess:
    d= sess.run(c, feed_dict={a:ones_array, b:ones_array})
# print(d)





# -------------END----------------------