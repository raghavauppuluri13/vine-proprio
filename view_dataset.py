import argparse
import os
import open3d as o3d
import cv2
import datetime
from util import PlyToState, update_points

import yaml
from pathlib import Path

from visualize import visualize_pcd, Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg-path", help="path to config file", type=str, default="./config/default.yml",
    )
    parser.add_argument(
        "-dataset-name", help="dataset name", type=str, required=True
    )

    args = vars(parser.parse_args())
    with open(args["cfg_path"], "r") as f:
        params = yaml.safe_load(f)

    dataset_folder = Path("{}/{}".format(params['data']['data_dir'],args['dataset_name']))
    kinect_pcd_folder = dataset_folder / 'kinect_pcd'
    kinect_rgb_folder = dataset_folder / 'kinect_rgb'
    vine_rgb_folder = dataset_folder / 'vine_rgb'

    dataset_size = len(list(vine_rgb_folder.glob('*')))

    pcd = o3d.geometry.PointCloud()

    state_dim = 36
    to_state = PlyToState(state_dim)

    v = Visualizer(state_dim)

    pcds = []

    try:
        for i in range(dataset_size):
            kinect_rgb = cv2.imread(str(kinect_rgb_folder / f'{i}.jpeg'))
            vine_rgb = cv2.imread(str(vine_rgb_folder / f'{i}.jpeg'))
            pcd = o3d.io.read_point_cloud(str(kinect_pcd_folder / f'{i}.ply'))
            pcds.append(pcd)
            state = to_state(pcd)
            print("sample",i)
            v.step(state, vine_rgb, kinect_rgb)
            #o3d.visualization.draw_geometries(pcds)
    except KeyboardInterrupt:
        pass
    v.save()
