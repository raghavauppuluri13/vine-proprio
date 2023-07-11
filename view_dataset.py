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
    pcd_folder = dataset_folder / 'pcd'
    img_folder = dataset_folder / 'img'

    dataset_size = len(list(img_folder.glob('*')))

    pcd = o3d.geometry.PointCloud()

    state_dim = 36
    to_state = PlyToState(state_dim)

    v = Visualizer(state_dim)

    pcds = []

    try:
        for i in range(dataset_size):
            img = cv2.imread(str(img_folder / f'{i}.jpeg'))
            pcd = o3d.io.read_point_cloud(str(pcd_folder / f'{i}.ply'))
            pcds.append(pcd)
            state = to_state(pcd)
            print(state)
            print("sample",i)
            v.step(state, img)
            #o3d.visualization.draw_geometries(pcds)
    except KeyboardInterrupt:
        pass
    v.save()
