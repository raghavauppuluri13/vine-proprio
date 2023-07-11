from torch import nn
import torch
import open3d as o3d
import cv2
import numpy as np

def crop_pcd(pcd, R, t, scale, bbox_params, visualize=False):
    # pretranslate
    T = np.eye(4)
    T[:3,3] = t
    pcd.transform(T)

    # rotate
    T = np.eye(4)
    T[:3,:3] = R
    pcd.transform(T)

    # bbox
    corners = get_centered_bbox(*bbox_params)
    corners *= scale
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))

    # Crop the point cloud
    cropped_pcd = pcd.crop(aabb)
    if visualize:
        visualize_pcds([cropped_pcd],frames=list(corners))

    return cropped_pcd

def get_centered_bbox(delta_y, x_pos, x_neg, z_pos, z_neg):
    x_pts = [[x_pos,0,0], [x_neg,0,0]]
    y_pts = [[0,delta_y,0], [0,-delta_y,0]]
    z_pts = [[0,0,z_pos], [0,0,z_neg]]

    pts = []
    for x in x_pts:
        for y in y_pts:
            for z in z_pts:
                pt = np.array([x,y,z]).sum(axis=0)
                pts.append(pt)
    pts = np.array(pts)
    return pts

def unit(v):
    return v / np.linalg.norm(v)

def visualize_pcds(pcds,frames=[]):

    # Add points to the line set
    for p in pcds:
        p.paint_uniform_color(list(np.random.uniform(0,1,3)))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
    )
    pcds.append(frame)

    for frame in frames:
        pcds.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100,  # specify the size of coordinate frame
                origin=list(frame)
            )
        )

    # Get the camera parameters of the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for pcd in pcds:
        vis.add_geometry(pcd)

    ctr = vis.get_view_control().convert_to_pinhole_camera_parameters()

    # Set the center of the viewport to the origin
    ctr.extrinsic = np.eye(4)
    vis.get_view_control().convert_from_pinhole_camera_parameters(ctr)

    # Update the visualization window
    vis.run()
    vis.destroy_window()

def rotate(v, angle):
    return np.linalg.norm(v) * np.array([np.cos(angle),np.sin(angle)])

def update_points(init_points, state: np.ndarray):
    curr_points = init_points.copy()
    for i in range(1, init_points.shape[0]):
        # Compute the angles between consecutive vectors
        v = curr_points[i] - init_points[i-1]
        curr_points[i] = curr_points[i-1] + rotate(v, state[i-1])
    return curr_points

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
    distance_threshold: int = 10
    ransac_iter: int = 1000
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def __call__(self,sample: o3d.geometry.PointCloud):

        pts_npy = np.asarray(sample.points)

        # 2d polynomial fit

        #TODO: invert negative in collect_dataset function instead of here
        z = np.polyfit(pts_npy[:,0],-pts_npy[:,1],3)

        state_points_dim= self.state_dim + 1
        state_pts = np.zeros((state_points_dim,3))

        x = np.linspace(0, np.max(pts_npy[:,0]), state_points_dim)
        p = np.poly1d(z)
        y = p(x)

        state_pts[:,0] = x
        state_pts[:,1] = y

        line_viz_pcd = o3d.geometry.PointCloud()
        line_viz_pcd.points = o3d.utility.Vector3dVector(state_pts)

        #visualize_pcds([line_viz_pcd,sample])

        # Compute the state_vectors between consecutive points
        state_pts = state_pts[:,:2]
        state_vectors = np.diff(state_pts, axis=0)
        state_angles = np.zeros(state_vectors.shape[0])

        for i in range(state_vectors.shape[0]):
            # Compute the angles between consecutive vectors
            v = state_vectors[i]
            v_u = v / np.linalg.norm(v)
            state_angles[i] = np.arctan(v_u[1]/v_u[0])
        return state_angles

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
