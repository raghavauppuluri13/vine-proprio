from torch import nn
import torch
import open3d as o3d
import cv2
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def obj_center_crop(im: cv2.Mat, desired: tuple = None):
    inds = np.argwhere(im != 0)
    max_p = np.max(inds, axis=0)
    min_p = np.min(inds, axis=0)
    centered = im[min_p[0] : max_p[0], min_p[1] : max_p[1]]
    if desired is not None:
        delta_w = desired[1] - centered.shape[1]
        delta_h = desired[0] - centered.shape[0]
        top, bottom = max(delta_h // 2, 0), max(delta_h - (delta_h // 2), 0)
        left, right = max(delta_w // 2, 0), max(delta_w - (delta_w // 2), 0)
        color = [0, 0, 0]
        centered = cv2.copyMakeBorder(
            centered, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        center = (centered.shape[0] / 2, centered.shape[1] / 2)
        x = center[1] - desired[1] / 2
        y = center[0] - desired[0] / 2

    centered = centered[int(y) : int(y + desired[0]), int(x) : int(x + desired[1])]
    return centered

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

class RandomSaturation(object):
    scale: float = 1 # (1.0-3.0)

    def __init__(self, scale):
        self.scale = scale

    def __call__(self,sample: torch.Tensor):
        im = sample.numpy()
        # Convert the image to HSV
        hsv_img = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

        # Increase the saturation
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * self.scale, 0, 255)

        # Convert back to BGR
        adjusted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return torch.from_numpy(adjusted_img)

class PlyToState(object):
    ransac_n: int = 3
    distance_threshold: int = 5
    ransac_iter: int = 1000
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def __call__(self,sample: o3d.geometry.PointCloud):

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
        return torch.from_numpy(state_angles)

class RandomContrast(object):
    alpha: float = 1.0 # (1.0-3.0)

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, sample: torch.Tensor):
        img = sample.numpy()
        adjusted_img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=0)
        return torch.from_numpy(adjusted_img)

class RandomGaussianNoise(object):
    mean: float = 0.0
    var: float = 1.0

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample: torch.Tensor):
        img = sample.numpy()
        sigma = np.sqrt(self.var)
        gaussian_noise = np.random.normal(self.mean, sigma, img.shape)
        noisy_img = img + gaussian_noise
        return torch.from_numpy(noisy_img)
