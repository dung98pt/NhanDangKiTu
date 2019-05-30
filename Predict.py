import numpy as np
import cv2
from skimage.feature import hog
from skimage import color
from sklearn.externals import joblib
from img_crop_resize import anhtest
from detect import Toa_Do
knn = joblib.load('./knn_model_full.pkl')
def feature_extraction(image):
    featureslist = []
    listhogdf = []
    training_digit = color.rgb2gray(image)
    featureslist.append(training_digit)
    for feature in featureslist:
        df = hog(feature, orientations=8,pixels_per_cell=(4,4), cells_per_block=(5,5))
        listhogdf.append(df)
    hogfeatures = np.array(listhogdf, 'float64')
    return hogfeatures
def predict(df):
    predict = knn.predict(df.reshape(1,-1))[0]
    predict = int(predict)
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    return predict, predict_proba[0][predict]

image = cv2.imread("liner.png",0)
# toado: list chứa tọa độ của các kí tự
image, toado = Toa_Do.out_put(image)
# Sắp xếp lại thứ tự
toado = sorted(toado, key=lambda a_entry:a_entry[2])
# Chứa các kí tự sau khi cắt
digits = []
for i in toado:
    # cắt từng kí tự
    img_one = image[i[0]:i[1] , i[2]:i[3]]
    digits.append(anhtest.img_resize(img_one))
# Tiến hành nhận dạng
hogs = list(map(lambda x: feature_extraction(x), digits))
predictions = list(map(lambda x: predict(x), hogs))


predictions = np.array(predictions, 'int')
output = []
for i in range(len(predictions)):
    if(predictions[i][0]<10):
        predictions[i][0]+=48
    elif(predictions[i][0]<36):
        predictions[i][0]+=55
    else:
        predictions[i][0]+=61
for i in range(len(predictions)):
    output.append([chr(predictions[i][0]),predictions[i][1]*100])
print(output)






















