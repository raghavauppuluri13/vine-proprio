from typing import *
from scipy.spatial import Delaunay
import math
from typing import Dict, Tuple
import random
import yaml
import argparse
import pathlib
from pathlib import Path

import open3d as o3d
import os
import numpy as np
import cv2

from visualize import visualize_pcd, visualize_line

import open3d as o3d

np.random.seed(10)

def preprocess_pcd(pcd, distance_threshold):
    # Fit a plane to the point cloud using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)

    # Extract inliers and outliers
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud])
    visualize_pcd(inlier_cloud)

    cloud_npy = np.asarray(inlier_cloud.points)

    # Proj all points to plane_model
    plane_model_unit_v = plane_model[:3] / np.linalg.norm(plane_model)
    plane_pt = np.zeros(3)
    plane_pt[2] = -plane_model[3] / plane_model[2]
    points_v = cloud_npy - plane_pt
    proj_dist = np.dot(points_v, plane_model_unit_v)
    proj_dist = proj_dist[...,np.newaxis]
    cloud_npy = cloud_npy - proj_dist * plane_model_unit_v

    # rotate plane to align with x axis
    base = np.eye(3)
    base[:,0] = plane_model_unit_v
    Q,R = np.linalg.qr(base)

    cloud_npy = (Q.T @ cloud_npy.T).T

    mean = np.mean(cloud_npy, axis=0)
    cloud_npy -= mean  # centering

    z = np.polyfit(pts_npy[:,1],pts_npy[:,2],3)

    state_pts = np.zeros((N,2))
    x = np.linspace(np.min(pts_npy[:,1]),np.max(pts_npy[:,1]),N)
    p = np.poly1d(z)
    y = p(x)

    state_pts[:,1] = x
    state_pts[:,2] = y

    state_pts_pcd.points = o3d.utility.Vector3dVector(state_pts)
    inlier_cloud.points = o3d.utility.Vector3dVector(cloud_npy)
    return inlier_cloud

def get_pcd():
    data = Path('../../mount/data/dataset_06-29-2023_18-32-54/pcd/50.ply')
    data = Path('../../cropped.ply')
    pcd = o3d.io.read_point_cloud(str(data.absolute()))  # replace with your point cloud file
    return pcd

if __name__ == "__main__":
    # Load your point cloud
    pcd = get_pcd()
    N = 50
    F = PlyToState(N)
    label = F(pcd)
    print(label)
