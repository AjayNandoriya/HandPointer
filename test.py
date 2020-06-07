""" Hand Tracker on Depth Image based on https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

"""

import os
import PIL
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
import json
import configparser
import csv

DATASET_BASE_DIR_NAME = r"D:\git\HandPointer\dataset"


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


def check_velocity():
    # json_file_name = os.path.join(DATASET_BASE_DIR_NAME, "result.json")
    # if os.path.isfile(json_file_name):
    #     with open(json_file_name, "r", encoding='utf8') as fid:
    #         datasets_info = json.load(fid)
    
    # dataset = datasets_info[id]
    csv_file_names = ['D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#fast_circles.csv'
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#gestures_two_hands.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#gestures_two_hands_swap.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#sequence_closed_hand.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#sequence_open_hand.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds325#sequence_small_shapes.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#circle_ccw.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#circle_ccw_far.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#circle_ccw_hand.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#circle_sequence.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#multiple_shapes_1.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#rectangle_ccw.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#rectangle_cw.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#star.csv',
                        'D:\git\HandPointer\dataset\#git#ajay#DepthSensing#dataset#ds536#zigzag.csv',
                        'D:\git\HandPointer\dataset\#git#HandPointer#dataset#ds325#fast_circles.csv']
    csv_file_name = os.path.join(r"D:\git\HandPointer\dataset", "#git#ajay#DepthSensing#dataset#ds325#fast_circles.csv")
    csv_file_name = csv_file_names[13]
    trajectory_data = np.zeros((800, 3))
    with open(csv_file_name, "r") as fid:
        reader = csv.reader(fid)
        header = next(reader)
        for row in reader:
            if len(row)<4:
                continue
            file_id = int(row[0])
            x = int(row[1])
            y = int(row[2])
            d = int(row[3])
            trajectory_data[file_id,0] = x
            trajectory_data[file_id,1] = y
            trajectory_data[file_id,2] = d

    velocity = np.zeros((800, 2))
    step = 5
    velocity[step:,0] = trajectory_data[step:,0] - trajectory_data[:-step,0]
    velocity[step:,1] = trajectory_data[step:,1] - trajectory_data[:-step,1]
    velocity_norm = np.linalg.norm(velocity, axis=1) 
    stop = (velocity_norm < 5).astype(np.uint8)
    plt.subplot(311)
    plt.plot(trajectory_data[:,0],'b')
    plt.plot(trajectory_data[:,1],'r')
    plt.plot(stop*100, 'g'), plt.title("xy")
    plt.subplot(312), plt.plot(velocity[:,0],'b')
    plt.plot(velocity[:,1],'r'), plt.title("velocity xy")
    plt.subplot(313), plt.plot(velocity_norm,'b'), plt.title("velocty norm")
    plt.plot(stop*100, 'g')
    plt.show()
    

def get_datasets():
    datasets = [
        r"ds325\fast_circles",
        r"ds325\gestures_two_hands",
        r"ds325\gestures_two_hands_swap",
        r"ds325\sequence_closed_hand",
        r"ds325\sequence_open_hand",
        r"ds325\sequence_small_shapes",
        r"ds536\circle_ccw",
        r"ds536\circle_ccw_far",
        r"ds536\circle_ccw_hand",
        r"ds536\circle_sequence",
        r"ds536\multiple_shapes_1",
        r"ds536\rectangle_ccw",
        r"ds536\rectangle_cw",
        r"ds536\star",
        r"ds536\zigzag",
        ]
    datasets = [os.path.join(DATASET_BASE_DIR_NAME, dataset) for dataset in datasets]
    return datasets

datasets_info = [{
    "base_dir_name" : r"D:\git\HandPointer\dataset\ds325\gestures_two_hands_swap",
    "max_file_count" : 600,
    "init_frame_id" :  50
}]

def calc_tajectory(file_id, loc, img_d, img_c):
    # x = int(bbox[0] + bbox[2])//2
    # y = int(bbox[1] + bbox[3])//2
    x = int(loc[0])
    y = int(loc[1])
    depth = img_d[y,x]
    confidence = img_c[y, x]
    trajectory = {
            "file_id" : file_id,
            "finger_tip" : {
                "x": x,
                "y": y,
            },
            "depth": depth,
            "confidence": confidence
        }
    return trajectory
        
def create_video_from_results():
    dataset_dir_names = get_datasets()
    
    for dataset_dir_name in dataset_dir_names:
        camera_file_name = os.path.join(os.path.dirname(dataset_dir_name), "camera_parameters.txt")
        mtx, dist, newcameramtx = read_camera_parameter(camera_file_name)
    
        video_file_name = dataset_dir_name.replace(DATASET_BASE_DIR_NAME,"")[1:]
        video_file_name = video_file_name.replace("\\", "_") + "_result.avi"
        video_file_name = os.path.join(DATASET_BASE_DIR_NAME, video_file_name)
        out = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 60, (320, 240))

        file_names = glob.glob(os.path.join(dataset_dir_name, "*_result.png"), recursive=True)
        for file_name in file_names:
            if not os.path.isfile(file_name):
                continue
            img = np.array(PIL.Image.open(file_name))
            img = img[:,:,::-1]
            img = cv2.undistort(img, mtx, dist, None, newcameramtx)
            
            out.write(img)
        out.release()

def create_video():
    dataset_dir_names = get_datasets()
    
    for dataset_dir_name in dataset_dir_names:
        camera_file_name = os.path.join(os.path.dirname(dataset_dir_name), "camera_parameters.txt")
        mtx, dist, newcameramtx = read_camera_parameter(camera_file_name)
    
        video_file_name = dataset_dir_name.replace(DATASET_BASE_DIR_NAME,"")[1:]
        video_file_name = video_file_name.replace("\\", "_") + "_depth.avi"
        video_file_name = os.path.join(DATASET_BASE_DIR_NAME, video_file_name)
        out = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'DIVX'), 60, (320, 240))

        file_names = glob.glob(os.path.join(dataset_dir_name, "*_depth.tiff"), recursive=True)
        for file_name in file_names:
            confidence_file_name = file_name.replace("depth", "confidence")
            if not os.path.isfile(file_name) or not os.path.isfile(confidence_file_name):
                continue
            img_d = np.array(PIL.Image.open(file_name)).astype(np.float)*0.1
            img_c = np.array(PIL.Image.open(confidence_file_name)).astype(np.float)*0.1
            
            img_d = np.clip(img_d, 0, 255).astype(np.uint8)
            img_c = np.clip(img_c, 0, 255).astype(np.uint8)
            
            img_d = cv2.undistort(img_d, mtx, dist, None, newcameramtx)
            img_c = cv2.undistort(img_c, mtx, dist, None, newcameramtx)

            img_out = np.zeros((*img_d.shape, 3), dtype=np.uint8)
            img_out[:,:, 0] = img_c.astype(np.uint8)
            img_out[:,:, 1] = img_d.astype(np.uint8)
            img_out[:,:, 2] = img_d.astype(np.uint8)
            out.write(img_out)
        out.release()

def read_camera_parameter(file_name):
    # file_name = r"D:\git\HandPointer\dataset\ds325\camera_parameters.txt"
    config = configparser.ConfigParser()
    config.read(file_name)
    data = {}
    for key in config['camera']:
        data[key] = float(config['camera'][key])

    mtx = np.eye(3)
    mtx[0, 0] = data['focal_x']
    mtx[1, 1] = data['focal_y']
    mtx[0, 2] = data['center_x']
    mtx[1, 2] = data['center_y']

    dist = (data['k1'], data['k2'], data['p1'], data['p2'], data['k3'])

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (320, 240), 1, (320, 240))

    # print(mtx, dist)
    # print(newcameramtx, roi)
    
    return mtx, dist, newcameramtx

def get_depth(img_d, img_c, bbox):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[0] + bbox[2])
    y2 = int(bbox[1] + bbox[3])
    img_d_roi = img_d[y1:y2, x1:x2]
    img_c_roi = img_c[y1:y2, x1:x2]

    min_val, max_c_val, min_loc, max_c_loc = cv2.minMaxLoc(img_c_roi)
    mask = (img_c_roi > max_c_val*0.3).astype(np.uint8)*255
    min_d_val, max_d_val, min_loc, max_d_loc = cv2.minMaxLoc(img_d_roi, mask)
    max_d_loc = (x1+max_d_loc[0], y1 + max_d_loc[1] )
    cv2.imshow("cropped_mask", mask)
    return max_d_val, max_c_val, max_d_loc


def test_dataset(id=0, n_traj=2):
    DEPTH_TH = 30
    tracker_type = 'BOOSTING'
    trackers = []
    for traj_id in range(n_traj):
        trackers.append(cv2.TrackerBoosting_create())

    json_file_name = os.path.join(DATASET_BASE_DIR_NAME, "result.json")
    if os.path.isfile(json_file_name):
        with open(json_file_name, "r", encoding='utf8') as fid:
            datasets_info = json.load(fid)
    
    dataset = datasets_info[id]
    init_frame_id = dataset['init_frame_id']
    dataset_dir_name = dataset['base_dir_name']
    max_file_count = dataset['max_file_count']
    dataset['trajectories'] = {}

    print( dataset['base_dir_name'])
    camera_file_name = os.path.join(os.path.dirname(dataset_dir_name), "camera_parameters.txt")
    mtx, dist, newcameramtx = read_camera_parameter(camera_file_name)

    hsv = plt.cm.get_cmap('hsv', max_file_count)

    trajectory_img = np.zeros((240, 320, 3), dtype=np.uint8)
    for file_id in range(max_file_count):

        depth_file_name = os.path.join(dataset_dir_name, r"{0:06d}_depth.tiff".format(file_id))
        confi_file_name = os.path.join(dataset_dir_name, r"{0:06d}_confidence.tiff".format(file_id))

        if not os.path.isfile(depth_file_name) or not os.path.isfile(confi_file_name):
            print( " file not found:", depth_file_name)
            continue
        img_d = np.array(PIL.Image.open(depth_file_name)).astype(np.float)
        img_c = np.array(PIL.Image.open(confi_file_name)).astype(np.float)

        img_d = 2500 - img_d
        img_d = cv2.undistort(img_d, mtx, dist, None, newcameramtx)
        img_c = cv2.undistort(img_c, mtx, dist, None, newcameramtx)


        img_d_norm = np.clip(img_d*0.1,0, 255).astype(np.uint8)
        img_d_norm = cv2.cvtColor(img_d_norm, cv2.COLOR_GRAY2BGR)
        # # Define an initial bounding box
        # bbox = (287, 23, 86, 320)
        # Uncomment the line below to select a different bounding box
        if file_id < init_frame_id:
            continue
        elif file_id == init_frame_id:
            for traj_id in range(n_traj):
                bbox_key = 'bbox{0}'.format(traj_id+1) 
                if bbox_key in dataset:
                    dataset[bbox_key] = [int(val) for val in dataset[bbox_key]]
                    bbox = tuple(dataset[bbox_key])
                else:
                    bbox = cv2.selectROI(img_d_norm, False)
                print("{0}: ".format(bbox_key), bbox)
                dataset[bbox_key] = bbox
                depth, confidence, max_d_loc = get_depth(img_d, img_c, bbox)
                img_d_norm_mask = (img_d_norm > depth*0.1 - DEPTH_TH).astype(np.uint8)*255 
                img_d_norm_masked = cv2.bitwise_and(img_d_norm, img_d_norm_mask)
                ok = trackers[traj_id].init(img_d_norm_masked, bbox)
                traj = calc_tajectory(file_id, max_d_loc, img_d, img_c)
                traj_key = "traj_{0}".format(traj_id + 1)
                dataset['trajectories'][traj_key] = [traj]
                
            # # display log image
            # cv2.imshow("img_d_norm", img_d_norm)
            # cv2.imshow("img_d_norm_masked", img_d_norm_masked)
            # cv2.waitKey(0)

            continue

        # Start timer
        timer = cv2.getTickCount()

        # depth = get_depth(img_d, img_c, bbox)
        
        # # display log image
        # cv2.imshow("img_d_norm", img_d_norm)
        # cv2.imshow("img_d_norm_masked", img_d_norm_masked)
        # cv2.waitKey(0)



        oks = []
        bboxes = []
        for traj_id in range(n_traj):
            depth = dataset['trajectories'][traj_key][-1]['depth']
            img_d_norm_mask = (img_d_norm > depth*0.1 - DEPTH_TH).astype(np.uint8)*255
            img_d_norm_masked = cv2.bitwise_and(img_d_norm, img_d_norm_mask)
            ok, bbox = trackers[traj_id].update(img_d_norm_masked)
            oks.append(ok)
            bboxes.append(bbox)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if all(oks):
            # Tracking success
            for traj_id in range(n_traj):
                p1 = (int(bboxes[traj_id][0]), int(bboxes[traj_id][1]))
                p2 = (int(bboxes[traj_id][0] + bboxes[traj_id][2]), int(bboxes[traj_id][1] + bboxes[traj_id][3]))
                cv2.rectangle(img_d_norm_masked, p1, p2, (255, 0, 0), 2, 1)
                depth, confidence, max_d_loc1 = get_depth(img_d, img_c, bboxes[traj_id])
                cv2.drawMarker(img_d_norm_masked, max_d_loc1, (255, 0, 255))
            
                traj = calc_tajectory(file_id, max_d_loc1, img_d, img_c)
                traj_key = "traj_{0}".format(traj_id + 1)
                dataset['trajectories'][traj_key].append(traj)
                
                p1 = dataset['trajectories'][traj_key][-2]['finger_tip']
                p1 = (int(p1['x']), int(p1['y']))
                p2 = dataset['trajectories'][traj_key][-1]['finger_tip']
                p2 = (int(p2['x']), int(p2['y']))
                color_val = hsv(file_id)[:3]
                color_val = (color_val[0]*255, color_val[1]*255, color_val[2]*255)
                cv2.line(trajectory_img, p1,p2, color_val)
        else:
            # Tracking failure
            cv2.putText(img_d_norm_masked, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # Display tracker type on frame
        cv2.putText(img_d_norm_masked, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(img_d_norm_masked, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", img_d_norm_masked)
        cv2.imshow("Trajectory", trajectory_img)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    
        # get_local_minima(img_d, img_c)
    with open(json_file_name, "w", encoding='utf8') as fid:
        json.dump(datasets_info, fid, indent=4)

if __name__ == "__main__":
    create_video_from_results()
    # test_dataset(id=15, n_traj=1)
    # check_velocity()