import scipy.io as sio
from PIL import Image

mat_data = sio.loadmat(r"C:\Users\luxin\Desktop\NLOS+SPI\GIDC-main\GIDC-main\data.mat")
print(mat_data)
print(mat_data.keys())


#  dict_keys(['__header__', '__version__', '__globals__', 'measurements', 'patterns'])
#  __' '__这样的是文件本身自带的属性，设置的属性为 measurement 和 patterns

key_name_1 = list(mat_data.keys())[-2]
data1 = mat_data[key_name_1]
print(data1.shape)
print(data1[1][0])

key_name_2 = list(mat_data.keys())[-1]
data2 = mat_data[key_name_2]
print(data2.shape)
print(data2[0][0])
print(len(data2[0][0]))

sum = 0
print(data2[0].shape)
for i in range(len(data2)):
    for j in range(len(data2)):
        sum += data2[0][i][j]
print(sum)

pattern = mat_data['patterns'][:,:,1]
print(pattern.shape)
img = Image.fromarray(pattern, 'L')
# print(pattern)
# img.show()


B_r = mat_data['measurements'][0]
print(B_r)