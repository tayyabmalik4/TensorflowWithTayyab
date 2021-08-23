# (07)*******************Sparce Tensor in Tensorflow*****************************

# ----Definition-----Sparse tensor enable efficient storage and processing of tensor that contain a lot of zero values.
# ---Syntax:----tf.SparseTensor

# -----image is achully shaped in 0 or 1 numbers
# -----When we converd the program to image then 1 is showes the black color and 0 is showes the white color
# -----then we convert to bytes and kilo bites we we formula-----
# -----shape=(8,17),dtype=int32 -----Size=(8X17X32)/8------544bytes=0.53KB
# -----Size of 22 pixels=22X32=88bytes=0.08KB


# -----Definition----Tensor that contain mostly zero values called sparse tensor.
# -----Dense Tensor Definition-----Tensors that contain non-zero values called danse Tensor.

# ----When working tensor that contain a lot of zero values, it is important to store them in a space and time-efficient manner.
# ----Sparse tensor enable efficient storage and processing of tensors that contain a lot of zero values.
# ----Sparse tensors are used extensively in encoding schemes like TF-IDF as part of data pre-processing in NLP applications and for pre-processing images with a lot of dark pixels in computer vision.

# -----Currently, Sparde tensors in Tensorflow are encoded using the coordinate list(COO) format

# -----The COO encoding for sparse is comprised of:
# ///value: A 1D tensor with shape[N] containing all nonzero values.
# ///indices: A 2D tensor with shape[N, rank], containing the indices of the nonzero values.
# ///dense_shape: A 1D tensor with shape[rank], specifying the shape of the tensor.

# ----A nonzero value in the context of a tf.Sparse Tensor is a value that's not explicity encoded.

# ----Syntax: tf.sparse.SparseTensor(indices,values,dense_shape)

# ----importing libraries
import numpy as np
import tensorflow as tf

# ----Creating sparse tensor
st1=tf.SparseTensor(indices=[[0,3],[2,4]],values=[10,20],dense_shape=[3,10])
# print(st1)

# -----Creat sparse Tensor from Dense
# -----First of all creating array
np_array=np.array([[1,4,3,2],[2,1,3,7],[9,8,1,2],[4,9,2,1]])
# print(np_array)

# -----creating dense tensor
st2_fd=tf.sparse.from_dense(np_array)
# print(st2_fd)

# -----Now we Extract the Values,Indices &shape of sparse Tensor
# print(st2_fd.values)
# ----if we want to extract just numpy values than we use this functions
# print(st2_fd.values.numpy())
# ----if we want to extract numpy values and also just extract list then we use this method
# print(st2_fd.values.numpy().tolist())

# ----if we want to extract the indices than we use this method
# print(st2_fd.indices)
# print(st2_fd.indices.numpy())
# print(st2_fd.indices.numpy().tolist())

# ----if we want to extract the shape then we use this method
# print(st2_fd.dense_shape)
# print(st2_fd.dense_shape.numpy())
# print(st2_fd.dense_shape.numpy().tolist())

# ----Sparse Tensor to Dense----if most of the elements are nonzero. then the lensor is considered dense.
dt_fst=tf.sparse.to_dense(st2_fd)
# print(dt_fst)
# ----Now converting the numpy format 
# print(dt_fst.numpy())
# print(dt_fst.shape)
# print(dt_fst.dtype)

# ----Mathematical operation on Sparse Tensor
ad=tf.sparse.add(st2_fd,st2_fd)
# print(ad)
# ----Now converting sparse to dense
d=tf.sparse.to_dense(ad)
# print(d)
# ----Now we need just numpy array 
print(d.numpy())




# ----------------END------------------------