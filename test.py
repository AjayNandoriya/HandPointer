""" Hand Tracker on Depth Image based on https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

"""

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

    h, w = img_d.shape[:2]
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
    tracker_type = 'BOOSTING'
    tracker1 = cv2.TrackerBoosting_create()
    tracker2 = cv2.TrackerBoosting_create()

    init_frame_id = 50
    dataset_dir_name = r"D:\git\ajay\DepthSensing\HandPointer\dataset\ds325\gestures_two_hands_swap"
    for k in range(600):

        depth_file_name = os.path.join(dataset_dir_name, r"{0:06d}_depth.tiff".format(k))
        confi_file_name = os.path.join(dataset_dir_name, r"{0:06d}_confidence.tiff".format(k))

        if not os.path.isfile(depth_file_name) or not os.path.isfile(confi_file_name):
            continue
        img_d = np.array(PIL.Image.open(depth_file_name)).astype(np.float)
        img_c = np.array(PIL.Image.open(confi_file_name)).astype(np.float)

        img_d_norm = (img_d*0.1).astype(np.uint8)
        img_d_norm = cv2.cvtColor(img_d_norm, cv2.COLOR_GRAY2BGR)
        # # Define an initial bounding box
        # bbox = (287, 23, 86, 320)
        # Uncomment the line below to select a different bounding box
        if k < init_frame_id:
            continue
        elif k == init_frame_id:
            bbox = cv2.selectROI(img_d_norm, False)
            ok = tracker1.init(img_d_norm, bbox)
            bbox = cv2.selectROI(img_d_norm, False)
            ok = tracker2.init(img_d_norm, bbox)
            continue

        # Start timer
        timer = cv2.getTickCount()

        ok1, bbox1 = tracker1.update(img_d_norm)
        ok2, bbox2 = tracker2.update(img_d_norm)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok1 and ok2:
            # Tracking success
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(img_d_norm, p1, p2, (255, 0, 0), 2, 1)

            p1 = (int(bbox2[0]), int(bbox2[1]))
            p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(img_d_norm, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(img_d_norm, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Display tracker type on frame
        cv2.putText(img_d_norm, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(img_d_norm, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", img_d_norm)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # get_local_minima(img_d, img_c)


if __name__ == "__main__":
    test_dataset()
