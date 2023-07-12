import argparse
import os
import open3d as o3d
import cv2
import datetime
from util import PlyToState, update_points, obj_center_crop
from PIL import Image

import yaml
from pathlib import Path

from visualize import visualize_pcd, Visualizer
from eval import EvalProprioNet, Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg-path", help="path to config file", type=str, default="./config/default.yml",
    )
    parser.add_argument(
        "-dataset-name", help="dataset name", type=str, required=True
    )

    parser.add_argument(
        "-train-log-dir", help="if specified, run eval using the data in trainlog",type=str, default=None
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

    state_dim = 5
    to_state = PlyToState(state_dim)

    if args['train_log_dir'] is not None:
        train_log_dir = Path('.log') / args["train_log_dir"]
        print(train_log_dir)
        with open(train_log_dir / 'config.yml', "r") as f:
            eval_params = yaml.safe_load(f)

        state_dim = eval_params['data']['state_dim']
        input_dim = eval_params['data']['image_size']
        compute_type = eval_params['eval']['compute_type']
        model_path = train_log_dir / 'chpt_final.pth'
        eval_ = EvalProprioNet(model_path=str(model_path.absolute()),
                            state_dim=state_dim,
                            input_dim=input_dim,
                            compute_type=compute_type)

    v = Visualizer(state_dim)

    e = Evaluator()

    pcds = []

    try:
        for i in range(dataset_size):
            kinect_rgb = cv2.imread(str(kinect_rgb_folder / f'{i}.jpeg'))
            new_shape = list(kinect_rgb.shape[:2])
            print(new_shape)
            new_shape[0] = new_shape[0] / 2
            new_shape[1] = new_shape[1] / 4
            kinect_rgb = obj_center_crop(kinect_rgb,new_shape)
            vine_rgb = cv2.imread(str(vine_rgb_folder / f'{i}.jpeg'))

            pcd = o3d.io.read_point_cloud(str(kinect_pcd_folder / f'{i}.ply'))
            pcds.append(pcd)
            state = to_state(pcd)
            if args['train_log_dir'] is not None:
                pred_state = eval_.run(vine_rgb)
                v.step(state, vine_rgb, kinect_rgb,pred_state=pred_state)
                e.add_sample(state,pred_state)
            else:
                v.step(state, vine_rgb, kinect_rgb)

            print("sample",i)
            #o3d.visualization.draw_geometries(pcds)
    except KeyboardInterrupt:
        pass
    print("RMSE: ", e.get_rmse())
    v.save()
