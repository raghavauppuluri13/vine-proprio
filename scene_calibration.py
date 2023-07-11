import numpy as np
import os
import cv2
import time
import open3d as o3d
import argparse
import datetime

from util import visualize_pcds, unit, get_centered_bbox, crop_pcd

EVAL_VINE_LEN =  390.02564018279617
EVAL_R = np. array([[ 0.11362603,  0.94958899, -0.29218123],
                     [-0.92689379,  0.20720155,  0.31294636],
                     [ 0.35771082,  0.23526211,  0.90371163]])
EVAL_t = np. array([  -22., 275.,  -506.])

EVAL_BBOX_PARAMS = [0.5,1,0,0.2,-0.2]

def get_rot_mat_from_basis(b1,b2,b3):
    A = np.eye(3)
    A[:,0] = b1
    A[:,1] = b2
    A[:,2] = b3
    return A.T

def get_calibration(pcd):
    print("")
    print(
        "1) Please pick 4 points using [shift + left click].\n \
        your 1st point should be the origin point \n \
        and your 2nd point should be along the x-axis on the vine robot when completely straight \n \
        and your 3rd point should be another point on the vine robot not on the x-axis \n \
        and your 4th point should be at the end of the vine robot "
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    points_idx = vis.get_picked_points()
    pcd_npy = np.asarray(pcd.points)


    vine_end = pcd_npy[points_idx[-1]]


    frame_pts = pcd_npy[points_idx[:3]]

    # compute basis vectors
    xaxis = unit(frame_pts[1] - frame_pts[0])
    v_another = unit(frame_pts[2] - frame_pts[0])
    zaxis = unit(np.cross(xaxis, v_another))
    yaxis = unit(np.cross(zaxis, xaxis))

    R = get_rot_mat_from_basis(xaxis, yaxis, zaxis)
    t = -frame_pts[0]

    # compute length of vine robot

    vine_len = np.linalg.norm(vine_end - frame_pts[0])

    return R, t, vine_len


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pcd_path", help="path to point cloud ply file", type=str, default="./scene.ply",
    )
    parser.add_argument(
        "-eval", help="view constants",action='store_true'
    )

    args = vars(parser.parse_args())
    pcd = o3d.io.read_point_cloud(args['pcd_path'])

    if args['eval']:
        vine_len = EVAL_VINE_LEN
        R = EVAL_R
        t = EVAL_t
    else:
        R,t,vine_len = get_calibration(pcd)

    cropped_pcd = crop_pcd(pcd,R,t,vine_len,EVAL_BBOX_PARAMS,visualize=True)

    print("vine_len = ", vine_len)
    print("R = np.", repr(R))
    print("t = np.", repr(t))
    print("BBOX_PARAMS = np.", EVAL_BBOX_PARAMS)
