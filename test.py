import os
import PIL
import numpy as np
from matplotlib import pyplot as plt
import cv2
def get_local_minima(img_d, img_c):
    img_d[200:, :] = 10000
    scale = 1/8
    confidence_thrshold = 100
    morph_kernel = np.ones((9, 9), np.uint8)

    h,w = img_d.shape[:2]
    sh = int(h*scale)
    sw = int(w*scale)
    imgd_scaled = cv2.resize(img_d, (sh, sw))
    imgc_scaled = cv2.resize(img_c, (sh, sw))

    mask = imgc_scaled > confidence_thrshold

    fimgd = cv2.morphologyEx(imgd_scaled, cv2.MORPH_BLACKHAT, morph_kernel)
    fimg = np.multiply(fimgd, mask.astype(np.uint8))

    inv_mask = np.invert(mask)
    imgd_scaled[inv_mask] = 10000
    # imgd_scaled = np.multiply(imgd_scaled, mask.astype(np.uint8))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgd_scaled, mask.astype(np.uint8))

    imgd_scaled = imgd_scaled-400
    cimg = (imgd_scaled.clip(min=0, max=600)/5).astype(np.uint8)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2BGR)
    cimg = cv2.drawMarker(cimg, min_loc, (0, 0, 0))
    cimg = cv2.resize(cimg, (500, 500))

    cv2.imshow("dpeth", cimg)
    cv2.waitKey(1)
    # print(min_loc, min_val)
    # ax1 = plt.subplot(121)
    # plt.imshow(mask)
    # plt.subplot(122, sharex=ax1, sharey=ax1)
    # plt.imshow(cimg), plt.title("after top hat")
    # plt.show()

def test_dataset():
    for k in range(600):

        depth_file_name = r"D:\git\ajay\DepthSensing\dataset\ds325\sequence_open_hand\{0:06d}_depth.tiff".format(k)
        confi_file_name = r"D:\git\ajay\DepthSensing\dataset\ds325\sequence_open_hand\{0:06d}_confidence.tiff".format(k)

        if not os.path.isfile(depth_file_name) or not os.path.isfile(confi_file_name):
            continue
        img_d = np.array(PIL.Image.open(depth_file_name)).astype(np.float)
        img_c = np.array(PIL.Image.open(confi_file_name)).astype(np.float)

        get_local_minima(img_d, img_c)



if __name__ == "__main__":
    test_dataset()