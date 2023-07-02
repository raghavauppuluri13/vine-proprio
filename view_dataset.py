import argparse
import os
import open3d as o3d
import cv2
import datetime
import yaml
from pathlib import Path

def visualize_mesh(pcd):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
        origin=list(pcd.points[0])  # specify the origin of the frame
    )
    o3d.visualization.draw_geometries([pcd,frame])

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

    dataset_folder = Path("{}/{}".format(params['dataset']['data_dir'],args['dataset_name']))
    pcd_folder = dataset_folder / 'pcd'
    img_folder = dataset_folder / 'img'

    dataset_size = len(list(img_folder.glob('*')))

    pcd = o3d.geometry.PointCloud()

    try:
        for i in range(dataset_size):
            img = cv2.imread(str(img_folder / f'{i}.jpeg'))
            cv2.imshow('Image',img)
            cv2.waitKey(0)
            pcd += o3d.io.read_point_cloud(str(pcd_folder / f'{i}.ply'))
            print("sample",i)
        visualize_mesh(pcd)
    except KeyboardInterrupt:
        pass
