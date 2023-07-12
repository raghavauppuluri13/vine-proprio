import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

import open3d as o3d

from util import update_points

def visualize_line(pcd, line_points_pcd):
    # Add points to the line set
    pcd.paint_uniform_color([1, 0, 0])  # Paint pc red
    line_points_pcd.paint_uniform_color([0, 1, 0])  # Paint state red

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
    )
    # Visualize the points and line
    o3d.visualization.draw_geometries([line_points_pcd,frame])

def visualize_pcd(pcd):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100,  # specify the size of coordinate frame
        origin=list(pcd.points[0])  # specify the origin of the frame
    )
    o3d.visualization.draw_geometries([pcd,frame])

class Visualizer:
    def __init__(self, state_dim,
                         L=1,
                         fps=5,
                         show=False,
                         dpi=50,
                         render_factor=20.0):
        self.N = state_dim + 1
        self.L = L
        self.init_points = np.zeros((self.N,2),dtype=np.float32)
        self.init_points[:,0] = np.linspace(0, self.L, self.N)
        fig, self.ax = plt.subplots()

        self.show = show
        self.fps = fps
        self.dpi = dpi
        self.render_factor = render_factor
        self.frames = []
        self.point_set = []
        self.frame_i = 0

    def reset(self):
        self.frames = []
        self.frame_i = 0

    def step(self, state, frame, kinect_frame,pred_state=None):

        self.frames.append((frame, kinect_frame))
        self.frame_i += 1

        if pred_state is None:
            pred_state = np.zeros_like(state)

        new_points = update_points(self.init_points, state)
        pred_points = update_points(self.init_points, pred_state)

        self.point_set.append((new_points,pred_points))

        if self.show:
            self.ax.plot(new_points[:,0], new_points[:,1], color='blue')
            self.ax.scatter(new_points[:,0], new_points[:,1], color='blue')
            self.ax.pause(0.02)


    def save(self, path="dataset_render.gif"):
        """Saves rendered frames as gif
        source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

        Args:
            path (str, optional): _description_. Defaults to 'dataset_render.gif'.
        """
        frames = self.frames
        fig,axes = plt.subplots(1,2,
            figsize=(
                frames[0][0].shape[1] / self.render_factor,
                frames[0][0].shape[0] / self.render_factor,
            ),
            dpi=self.dpi,
        )
        ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (0, 0))
        ax3 = plt.subplot2grid((2, 2), (0, 1))

        # Set x-axis label
        ax1.set_xlabel('X-position (m)')

        # Set y-axis label
        ax1.set_ylabel('Y-position (m)')

        # Set x-axis limits
        ax1.set_xlim([0, self.L+1])

        # Set y-axis limits
        ax1.set_ylim([-1, 1])

        # set titles
        ax2.set_title("Vine Robot Embedded Camera Image")
        ax1.set_title("Vine Robot State Estimate")
        ax3.set_title("External Camera View")

        ax1.legend()
        # Add grid

        ax1.grid(True)
        ax2.grid(False)
        ax3.grid(False)
        ax2.axis('off')
        ax3.axis('off')


        vine_cam_patch = ax2.imshow(frames[0][0])
        kinect_cam_patch = ax3.imshow(frames[0][1])
        plot = ax1.plot(self.point_set[0][0][:,0], self.point_set[0][0][:,1], color='red', label="Ground Truth")[0]
        plot_scatter = ax1.plot(self.point_set[0][0][:,0], self.point_set[0][0][:,1], color='red',marker='o')[0]

        pred_plot = ax1.plot(self.point_set[0][1][:,0], self.point_set[0][1][:,1], color='green', label="Predicted")[0]
        pred_plot_scatter = ax1.plot(self.point_set[0][1][:,0], self.point_set[0][1][:,1], color='green',marker='o')[0]
        ax1.legend()

        plt.draw()

        def animate(i):
            vine_cam_patch.set_data(frames[i][0])
            kinect_cam_patch.set_data(frames[i][1])
            plot.set_data(self.point_set[i][0][:,0], self.point_set[i][0][:,1])
            plot_scatter.set_data(self.point_set[i][0][:,0], self.point_set[i][0][:,1])

            pred_plot.set_data(self.point_set[i][1][:,0], self.point_set[i][1][:,1])
            pred_plot_scatter.set_data(self.point_set[i][1][:,0], self.point_set[i][1][:,1])

        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames), interval=50
        )
        anim.save(path, writer="imagemagick", fps=self.fps)
