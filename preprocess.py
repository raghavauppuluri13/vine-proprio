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


import open3d as o3d

np.random.seed(10)

class PlyToState(object):
    ransac_n: int = 3
    distance_threshold: int = 5
    ransac_iter: int = 1000
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def __call__(self, sample: o3d.geometry.PointCloud):

        # Fit a plane to the point cloud using RANSAC
        plane_model, inliers = sample.segment_plane(distance_threshold=self.distance_threshold,
                                                ransac_n=self.ransac_n,
                                                num_iterations=self.ransac_iter)

        # Extract inliers and outliers
        inlier_cloud = sample.select_by_index(inliers)
        outlier_cloud = sample.select_by_index(inliers, invert=True)

        # Proj all points to plane
        pts_npy = np.asarray(inlier_cloud.points)

        plane_model_unit_v = plane_model[:3] / np.linalg.norm(plane_model)
        plane_pt = np.zeros(3)
        plane_pt[2] = -plane_model[3] / plane_model[2]
        points_v = pts_npy - plane_pt
        proj_dist = np.dot(points_v, plane_model_unit_v)
        proj_dist = proj_dist[...,np.newaxis]
        pts_npy = pts_npy - proj_dist * plane_model_unit_v

        # rotate plane to align with x axis
        base = np.eye(3)
        base[:,0] = plane_model_unit_v
        Q,R = np.linalg.qr(base)

        pts_npy = (Q.T @ pts_npy.T).T

        # Centering
        mean = np.mean(pts_npy, axis=0)
        pts_npy -= mean

        # 2d polynomial fit
        z = np.polyfit(pts_npy[:,1],pts_npy[:,2],3)

        state_pts = np.zeros((self.state_dim,2))

        # start farther away and come closer
        x = np.linspace(np.max(pts_npy[:,1]),np.min(pts_npy[:,1]),N)
        p = np.poly1d(z)
        y = p(x)

        state_pts[:,0] = x
        state_pts[:,1] = y

        # Compute the state_vectors between consecutive points
        state_vectors = np.diff(state_pts, axis=0)
        state_angles = np.zeros(state_vectors.shape[0])

        for i in range(1, state_vectors.shape[0]):
            # Compute the angles between consecutive vectors
            v1 = state_vectors[i]
            v2 = state_vectors[i-1]
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            state_angles[i] = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
         # TODO: the state seems to fluctate by 0.02 between calls for the same
         # pointcloud
        return state_angles


def visualize_pcd(points):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
    )
    o3d.visualization.draw_geometries([pcd,frame])

def visualize_line(pcd, line_points_pcd):
    # Add points to the line set
    pcd.paint_uniform_color([1, 0, 0])  # Paint pc red
    line_points_pcd.paint_uniform_color([0, 1, 0])  # Paint state red

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
    )
    # Visualize the points and line
    o3d.visualization.draw_geometries([line_points_pcd,frame])

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
