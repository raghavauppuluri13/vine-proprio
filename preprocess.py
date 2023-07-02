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

def visualize_line(points, line_points):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Create LineSet object
    line_set = o3d.geometry.LineSet()

    # Add points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 0, 0])  # Paint inliers red

    # Add points to the line set
    line_set.points = o3d.utility.Vector3dVector(line_points)

    # Add lines using indices
    lines = [[0, i] for i in range(1, len(line_points))]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Visualize the points and line
    o3d.visualization.draw_geometries([pcd, line_set])

def preprocess_point_cloud(pcd, distance_threshold):
    # Fit a plane to the point cloud using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=3,
                                             num_iterations=1000)

    # Extract inliers and outliers
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return inlier_cloud, outlier_cloud

def pca(pts):
    pts = pts.reshape(-1, 2).astype(np.float64)
    mv = np.mean(pts, 0).reshape(2, 1)
    pts -= mv.T
    w, v = np.linalg.eig(np.dot(pts.T, pts))
    w_max = np.max(w)
    w_min = np.min(w)
    col = np.where(w == w_max)[0]
    if len(col) > 1:
        col = col[-1]
    V_max = v[:, col]

    if V_max[0] > 0 and V_max[1] > 0:
        V_max *= -1

    col_min = np.where(w == w_min)[0]
    if len(col_min) > 1:
        col_min = col_min[-1]
    V_min = v[:, col_min]

    return V_max, V_min, w_max, w_min, mv

def moving_least_sq(mst: Dict[Tuple,float], h_dist):
    def collect(P, A):
        A.add(P)
        for (Pi_idx, Pj_idx), dist in mst.items():
            if np.abs(dist) < h_dist:
                return collect(Pj_idx,A)
    Pstar = 0
    A = set()
    collect(Pstar, A)

    return np.array(A)


def get_state(pcd: o3d.geometry.PointCloud,  N):
    # find bbox

    bbox = pcd.get_axis_aligned_bounding_box()
    print(bbox)

    points = pcd.points
    mean = np.mean(points, axis=0)
    points -= mean  # centering
    print(points)

    # Computing Delaunay
    tri = Delaunay(points)

    # Building the edges-distance map:
    edges_dist_map = {}
    for tr in tri.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]
            if (edge_idx1, edge_idx0) in edges_dist_map:
                continue  # already visited this edge from other side
            p0 = points[edge_idx0]
            p1 = points[edge_idx1]
            dist = np.linalg.norm(p1 - p0)
            edges_dist_map[(edge_idx0, edge_idx1)] = dist
    line_pts = moving_least_sq(edges_dist_map, 1)
    print(line_pts)


    return points[line_pts,:]

def get_pcd():
    data = Path('./data/dataset_06-29-2023_18-32-54/pcd/120.ply')

    pcd = o3d.io.read_point_cloud(str(data.absolute()))  # replace with your point cloud file
    return pcd

if __name__ == "__main__":
    # Load your point cloud
    pcd = get_pcd()

    # Preprocess the point cloud
    inlier_cloud, outlier_cloud = preprocess_point_cloud(pcd, distance_threshold=5)
    # Define the number of points
    N = 100
    line_pts = get_state(inlier_cloud, N)
    visualize_line(inlier_cloud, line_points)
