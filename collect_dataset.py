import numpy as np
import yaml
import os
import cv2
import time
import k4a
import open3d as o3d
import argparse
import datetime

from k4a._bindings.k4atypes import *

BOUNDING_BOX = np.array([
    [-190.74442147,-344.92555785,1400.        ],
    [-389.25557853,-325.07444215,1400.        ],
    [-349.25557853, 264.92555785, 760.        ],
    [  39.25557853, 235.07444215, 770.        ],
    [-190.75904138,-345.071757  ,1399.86430892],
    [-389.27019845,-325.22064129,1399.86430892],
    [-349.27019845, 264.77935871, 759.86430892],
    [  39.24095862, 234.928243  , 769.86430892]]
)

def crop(points, pcd):
    # Compute the center of the box
    center = points.mean(axis=0)

    # Compute the extent of the box (the length, width, and height)
    extent = points.max(axis=0) - points.min(axis=0)

    # Compute the rotation of the box. This step assumes that the points are ordered in a specific way.
    # If that's not the case, you'll need a more complex method to compute the rotation.
    R = np.array([
        points[1] - points[0],
        points[3] - points[0],
        points[4] - points[0],
    ]).T
    R /= np.linalg.norm(R, axis=0)

    # Create the oriented bounding box
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

    # Crop the point cloud
    return pcd.crop(obb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg_path", help="path to config file", type=str, default="./config/default.yml"
    )

    args = vars(parser.parse_args())
    with open(args["cfg_path"], "r") as f:
        params = yaml.safe_load(f)

    dataset_folder = datetime.datetime.now().strftime("{}/dataset_%m-%d-%Y_%H-%M-%S".format(params['dataset']['data_dir']))
    os.mkdir(dataset_folder)

    pcd_folder = dataset_folder + '/pcd'
    img_folder = dataset_folder + '/img'
    os.mkdir(pcd_folder)
    os.mkdir(img_folder)

    # Open Device
    vcap = cv2.VideoCapture("http://192.168.0.100:8000/stream.mjpg")
    device = k4a.Device.open()

    # Start Cameras
    device_config = DeviceConfiguration(
        color_format = EImageFormat.COLOR_BGRA32,
        color_resolution = EColorResolution.RES_720P,
        depth_mode = EDepthMode.NFOV_UNBINNED,
        camera_fps = EFramesPerSecond.FPS_30,
        synchronized_images_only = True,
        depth_delay_off_color_usec = 0,
        wired_sync_mode = EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec = 0,
        disable_streaming_indicator = False)

    device_config = k4a.DEVICE_CONFIG_BGRA32_1080P_WFOV_UNBINNED_FPS15
    device.start_cameras(device_config)

    # Get Calibration
    calibration = device.get_calibration(
        depth_mode=device_config.depth_mode,
        color_resolution=device_config.color_resolution)

    # Create Transformation
    transformation = k4a.Transformation(calibration)
    i = 0

    try:
        while(True):
            # Capture One Frame
            capture = device.get_capture(-1)
            _, rpi_img = vcap.read()

            # Get Point Cloud
            point_cloud = transformation.depth_image_to_point_cloud(capture.depth, k4a.ECalibrationType.DEPTH)

            # Save Point Cloud To Ascii Format File. Interleave the [X, Y, Z] channels into [x0, y0, z0, x1, y1, z1, ...]
            height, width, channels = point_cloud.data.shape
            xyz_data = point_cloud.data.reshape(height * width, channels)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_data)

            pcd = crop(BOUNDING_BOX,pcd)
            o3d.io.write_point_cloud(f"{pcd_folder}/{i}.ply", pcd)
            cv2.imwrite(f'{img_folder}/{i}.jpeg',rpi_img)
            print(f'sample {i} saved')
            i += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f'dataset {dataset_folder} finished!')

    # Stop Cameras
    device.stop_cameras()
