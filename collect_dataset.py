import numpy as np
import yaml
import os
import queue
import threading
import cv2
import time
import k4a
import open3d as o3d

import urllib
import argparse
import datetime

from multiprocessing import Process
from util import crop_pcd, visualize_pcds, unit

from k4a._bindings.k4atypes import *

PARAM_vine_len =  390.02564018279617
PARAM_R = np.array([[ 0.11362603,  0.94958899, -0.29218123],
                      [-0.92689379,  0.20720155,  0.31294636],
                      [ 0.35771082,  0.23526211,  0.90371163]])
PARAM_t = np.array([ -22.,  275., -506.])
PARAM_bbox = [0.5, 1, 0, 0.2, -0.2]


class Buffer:
    def __init__(self, max_size):
        self.queue = queue.LifoQueue()
        self.max_size = max_size

    def put(self, item):
        if self.queue.qsize() >= self.max_size:
            self.queue.get()
        self.queue.put(item)

    def get(self):
        return self.queue.get()

class KinectCapture:
    def __init__(self):
        self.device = k4a.Device.open()

        # Start Cameras
        self.device_config = DeviceConfiguration(
            color_format = EImageFormat.COLOR_BGRA32,
            color_resolution = EColorResolution.RES_720P,
            depth_mode = EDepthMode.NFOV_UNBINNED,
            camera_fps = EFramesPerSecond.FPS_30,
            synchronized_images_only = True,
            depth_delay_off_color_usec = 0,
            wired_sync_mode = EWiredSyncMode.STANDALONE,
            subordinate_delay_off_master_usec = 0,
            disable_streaming_indicator = False)

        self.device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_WFOV_UNBINNED_FPS15
        self.device.start_cameras(self.device_config)

        # Get Calibration
        self.calibration = self.device.get_calibration(
            depth_mode=self.device_config.depth_mode,
            color_resolution=self.device_config.color_resolution)

        # Create Transformation
        self.transformation = k4a.Transformation(self.calibration)

    def release(self):
        self.device.stop_cameras()

    def read(self):
        capture = self.device.get_capture(-1)
        # Get Point Cloud
        point_cloud = self.transformation.depth_image_to_point_cloud(capture.depth, k4a.ECalibrationType.DEPTH)

        # Save Point Cloud To Ascii Format File. Interleave the [X, Y, Z] channels into [x0, y0, z0, x1, y1, z1, ...]
        height, width, channels = point_cloud.data.shape
        xyz_data = point_cloud.data.reshape(height * width, channels)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_data)
        return pcd


def preprocess(pcd):
    global PARAM_R, PARAM_t, PARAM_vine_len, PARAM_bbox

    pcd = crop_pcd(pcd,PARAM_R,PARAM_t,PARAM_vine_len,PARAM_bbox)

    ransac_n: int = 3
    distance_threshold: int = 20
    ransac_iter: int = 1000
    # Fit a plane to the point cloud using RANSAC

    '''
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=ransac_iter)

    # Extract inliers and outliers
    inlier_cloud = pcd.select_by_index(inliers)

    # Apply DBSCAN clustering
    eps = 10  # The maximum distance between two samples for them to be considered as in the same cluster
    min_points = 10  # The minimum number of samples in a cluster
    clusters = np.array(inlier_cloud.cluster_dbscan(eps=eps, min_points=min_points))

    # Find the main cluster (the one with the most points)
    unique, counts = np.unique(clusters, return_counts=True)
    main_cluster_label = unique[np.argmax(counts)]

    # Remove all points not in the main cluster
    main_cluster_indices = np.where(clusters == main_cluster_label)[0]
    inlier_cloud = inlier_cloud.select_by_index(main_cluster_indices)
    '''

    plane_model_unit_v = np.array([0,0,1])

    # Proj all points to plane
    pts_npy = np.asarray(pcd.points)
    #pts_npy = np.append(pts_npy, [ORIGIN,XAXIS_PT],axis=0)

    plane_pt = np.zeros(3)
    #plane_pt[2] = -plane_model[3] / plane_model[2]
    vec_points = pts_npy - plane_pt
    proj_dist = np.dot(vec_points, plane_model_unit_v) # (N,3) dot (3,) = (N,)
    proj_dist = proj_dist[...,np.newaxis] # (N,1)

    # (N,3) - (N,1) * (3,) = (N,3) - (N,3) = (N,3)
    pts_npy = pts_npy - proj_dist * plane_model_unit_v

    pcd.points = o3d.utility.Vector3dVector(pts_npy)

    '''
    W = world frame
    B = base frame of vine robot

    we want R_B_W

    # all are orthogonal to each other, span R3
    # defined in W
    basis = { b1, b2, b3 }

    A = [b1,b2,b3]

    p_b = point in B
    p_w = point in w

    p_w = A mmul p_b
    > A.T mmul p_w = A.T mmul A mmul p_b # A.T equiv inv(A) when A is orthogonal
    > A.T mmul p_w = I mmul p_b
    > A.T mmul p_w = p_b
    > R_B_W = A.T mmul p_w

    # length-wise axis of vine robot is x-axis
    b1 = unit(pts_npy[-1] - pts_npy[-2])

    # vector perpendicular to plane is z-axis
    b3 = plane_model_unit_v
    b2 = unit(np.cross(b3,b1))

    # translate origin to base point
    T = np.eye(4)
    T[:3,3] = -pts_npy[-2]
    pcd.transform(T)

    # rotate frame to base frame
    T = np.eye(4)
    T[:3,0] = b1
    T[:3,1] = b2
    T[:3,2] = b3

    pcd.transform(T.T)
    '''

    #visualize_pcds([pcd])

    return pcd

def read_pcd(pcd_buffer, capture, stop_event):
    while not stop_event.is_set():
        pcd = capture.read()
        pcd_buffer.put(pcd)
        time.sleep(0.01)

def read_rgb(rgb_buffer, capture, stop_event):
    while not stop_event.is_set():
        ret, rgb = capture.read()
        if ret:
            rgb_buffer.put(rgb)
        time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg_path", help="path to config file", type=str, default="./config/default.yml"
    )
    parser.add_argument(
        "-scene-collect-only", help="just write scene.ply",action='store_true'
    )

    args = vars(parser.parse_args())
    with open(args["cfg_path"], "r") as f:
        params = yaml.safe_load(f)

    # Open Streams
    vcap = cv2.VideoCapture("http://192.168.0.100:8000/stream.mjpg")
    kinect_capture = KinectCapture()

    pcd_buffer = Buffer(10)
    rgb_buffer = Buffer(10)
    stop_event = threading.Event()

    # Start the threads for reading depth and RGB images
    pcd_thread = threading.Thread(target=read_pcd, args=(pcd_buffer, kinect_capture, stop_event))
    rgb_thread = threading.Thread(target=read_rgb, args=(rgb_buffer, vcap, stop_event))

    pcd_thread.start()
    rgb_thread.start()

    if args['scene_collect_only']:
        pcd = pcd_buffer.get()
        o3d.io.write_point_cloud(f"./scene.ply", pcd)
    else:
        # Create dataset folders
        dataset_folder = datetime.datetime.now().strftime("{}/dataset_%m-%d-%Y_%H-%M-%S".format(params['data']['data_dir']))
        os.mkdir(dataset_folder)

        pcd_folder = dataset_folder + '/pcd'
        img_folder = dataset_folder + '/img'
        os.mkdir(pcd_folder)
        os.mkdir(img_folder)

        i = 0

        try:
            while(True):
                rgb = rgb_buffer.get()
                pcd = pcd_buffer.get()
                pcd = preprocess(pcd)

                # Get Point Cloud
                o3d.io.write_point_cloud(f"{pcd_folder}/{i}.ply", pcd)
                cv2.imwrite(f'{img_folder}/{i}.jpeg',rgb)
                print(f'sample {i} saved')
                time.sleep(0.03)
                i += 1
        except KeyboardInterrupt:
            pass
        print(f'dataset {dataset_folder} finished!')

    stop_event.set()
    pcd_thread.join()
    rgb_thread.join()

    # Stop Cameras
    kinect_capture.release()
    vcap.release()
