import numpy as np
import yaml
import matplotlib
import argparse
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from eval import EvalProprioNet
import time

from visualize import Visualizer

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg_path", help="path to config file", type=str, default="./config/default.yml"
    )

    args = vars(parser.parse_args())
    with open(args["cfg_path"], "r") as f:
        params = yaml.safe_load(f)



    vcap = cv2.VideoCapture("http://192.168.0.100:8000/stream.mjpg")
    state_dim = params['data']['state_dim']
    input_dim = params['data']['image_size']
    compute_type = params['eval']['compute_type']
    model_path = params['eval']['model_path']
    eval_ = EvalProprioNet(model_path=model_path,
                           state_dim=state_dim,
                           input_dim=input_dim,
                           compute_type=compute_type)

    # compute equidistant points

    v = Visualizer(state_dim)

    i = 0

    try:
        while True:
            _, rpi_img = vcap.read()
            state = eval_.run(rpi_img)
            v.step(state, rpi_img)
            print(f'frame {i}')
            time.sleep(0.1)
            i += 1
    except KeyboardInterrupt:
        print('finished! saving!')
    v.save()



