import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import yaml


class STFTools:
    def __init__(self, yaml_path=None):
        self.configs = self.read_yaml(yaml_path)
        self.data_path = self.configs['DATA_PATH']
        self.imgSet_path = self.data_path + '/ImageSets'
        self.train_path = self.data_path + '/training'
        self.calib_path = self.train_path + '/calib'
        self.data_type = self.configs['DATA_TYPE']
        if self.data_type == 'stf':
            self.calib_file = self.calib_path + '/kitti_stereo_velodynehdl_calib.txt'   # stf using hdl32
        if self.data_type == 'kitti':
            print("Kitti data type not supported yet")
        with open(self.calib_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line or line == '\n':
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                values = list(map(float, value.strip().split()))
                if key == 'P0':
                    self.P0 = np.array(values).reshape(3, 4)
                elif key == 'P1':
                    self.P1 = np.array(values).reshape(3, 4)
                elif key == 'P2':
                    self.P2 = np.array(values).reshape(3, 4)
                elif key == 'P3':    
                    self.P3 = np.array(values).reshape(3, 4)
                elif key == 'R0_rect':
                    self.R0_rect = np.array(values).reshape(3, 3)
                elif key == 'Tr_velo_to_cam':    
                    self.Tr_velo_to_cam = np.array(values).reshape(3, 4)
                elif key == 'Tr_imu_to_velo':
                    self.Tr_imu_to_velo = np.array(values).reshape(3, 4)
                elif key == 'Tr_radar_to_cam':
                    self.Tr_radar_to_cam = np.array(values).reshape(3, 4)

        pass

    def read_yaml(self, path):
        with open(path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def read_velodyne_bin(self, bin_path):
        """
        Read velodyne bin file
        :param bin_path: path to the bin file
        :return: numpy array of shape (n, 4)
        """
        scan = np.fromfile(bin_path, dtype=np.float32)
        scan = scan.reshape((-1, 5))
        return scan
    
    
    

    

if __name__ == "__main__":
    yaml_path = "/home/sorodo/SLAM/dcp_Virconv/uitls/stf_cfg.yaml"
    stf = STFTools(yaml_path)
    print(stf.P0)
    scan = stf.read_velodyne_bin("/mnt/h/data/seeingthroughfog/training/velodyne/2018-02-03_20-48-35_00500.bin")
    print(scan)