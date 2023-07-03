import numpy as np
ghost = np.zeros((800, 1280))
print(ghost.shape)
a = ghost[:,0]
b = ghost[0,:]
print(a)
print(b)
print(len(a))
print(len(b))
# 输出的图像应该是一个长方形形状，高度H为800，长度W为1280
