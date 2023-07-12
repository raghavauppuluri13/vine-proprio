# vine-proprio
Estimate the proprioception of a [vine robot](https://www.youtube.com/watch?v=oRjFFgAZQnk) from an internal camera view

![vine-robot-setup](https://github.com/raghavauppuluri13/vine-proprio/assets/41026849/cdf8aa1e-53c0-4d9a-ad3a-12a831838bd0)

![eval_close](https://github.com/raghavauppuluri13/vine-proprio/assets/41026849/e6edfdb9-36bf-498b-a285-a9fe18b6bb1e)


### Structure

#### `scene_calibration.py`
- Given a raw scene point cloud, generate calibration parameters to crop the raw scene to just the vine robot (use them in `collect_dataset.py`)
#### `collect_dataset.py`
- Collect processed pointclouds from a Microsoft Azure Kinect pointcloud stream and RPI camera over http stream (TODO: can decrease latency of stream with multiprocessing)
#### `train.py`
- Supervised learning training setup using MSE Loss, logs and tensorboard outputs are stored
#### `model.py`
- defines simple ResNet for fixed feature extraction and trainable MLP final layers

#### `util.py`
- various preprocessing functions
#### `view.py`
- given a dataset and optionally a log folder from a training output, generates a timeseries gif with predicted state plots and camera views
#### `visualize.py`
- Matplotlib visualizer that outputs a gif of an external view, internal view, and estimated/groundtruth vine robot state
#### `data.py`
- defines structure of the proprioception dataset
#### `eval.py`
- defines an inference class and evaluation class to simplify model evaluation
