import numpy as np
import cv2
class anhtest:
    def img_crop(img):
        x = img
        doc = []
        ngang = []
        for i in range(x.__len__()):
            for j in range(x[0].__len__()):
                if (x[i][j] != 255):
                    doc.append(i)
        for j in range(x[0].__len__()):
            for i in range(x.__len__()):
                if (x[i][j] != 255):
                    ngang.append(j)
        
        a = doc[-1] - doc[0]
        b = ngang[-1] - ngang[0]

        img_crop = x[doc[0]:doc[-1], ngang[0]:ngang[-1]]
        return img_crop, a, b

    def img_resize(img_crop):
        a,b = img_crop.shape
        c = max(a, b)
        img_pad = np.zeros([c, c])
        if (a > b):
            img_pad[:, int((a - b) / 2): int((a + b) / 2)] = img_crop
        else:
            img_pad[int((b - a) / 2): int((a + b) / 2), : ] = img_crop
        img_resized = cv2.resize(src=img_pad, dsize=(20, 20))
        return img_resized

