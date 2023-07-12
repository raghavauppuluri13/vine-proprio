# vine-proprio

Estimate the state of a vine robot from an internal view

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
#### `data.py`
- defines structure of the proprioception dataset
#### `eval.py`
- defines an inference class and evaluation class to simplify model evaluation
