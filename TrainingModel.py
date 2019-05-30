import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import color
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

# Đọc vào ảnh training
path = "data_chu/train/"
folder = os.listdir(os.path.expanduser(path))
train = []
label = []

# đọc vào dữ liệu trainning
for i in range(len(folder)):
    PATH = path + folder[i] + "/"
    file = os.listdir(os.path.expanduser(PATH))
    k = i
    for y in file:
        training_digit_image = cv2.imread(PATH + y, 0)
        #chuyển ảnh xám
        training_digit = color.rgb2gray(training_digit_image)
        #biến đổi hog
        df = hog(training_digit_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(5, 5))
        train.append(df)
        label.append(k)
# reshape lại dữ liệu
train = np.array(train, 'float64')


# Đọc vào dữ liệu test
path_test = "data_chu/test/"
folder_test = os.listdir(os.path.expanduser(path_test))
test = []
label_test = []
for x in range(len(folder_test)):
    PATH = path_test + folder_test[x] + "/"
    file = os.listdir(os.path.expanduser(PATH))
    k = x
    for y in file:
        training_digit_image = cv2.imread(PATH + y, 0)
        # biến đổi ảnh xám
        training_digit = color.rgb2gray(training_digit_image)
        # biến đổi hog
        df = hog(training_digit_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(5, 5))
        test.append(df)
        label_test.append(k)
# reshape lại dữ liệu
test = np.array(test, 'float64')


# train using K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train, label)
# độ chính xác
model_score = knn.score(test, label_test)

# lưu lại mô hình đã training
joblib.dump(knn, './knn_model_full.pkl')
print(model_score)



