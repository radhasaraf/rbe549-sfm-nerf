import numpy as np
import cv2
import argparse
from utils.visualization_utils.py import *

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath',default='../data/Calibration_Imgs')
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")

    args = parser.parse_args()
    main(args)
