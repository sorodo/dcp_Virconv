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
        self.lidar_path = self.train_path + '/velodyne'
        self.image_path = self.train_path + '/image_2'
        self.label_path = self.train_path + '/label_2'
        
        self.lidar_files = os.listdir(self.lidar_path)

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
    
    def transform_one_point2Cam(self, point):
        """
        Transform matrix from velodyne to camera
        :param point: numpy array of shape (3,)
        """
        Tr = np.vstack((self.Tr_velo_to_cam, [0, 0, 0, 1]))
        homogenous_point = np.hstack((point, 1))
        point = np.dot(Tr, homogenous_point)
        return point[:3]

    def transform_points(self, points):
        """
        Transform matrix from velodyne to camera
        :param points: numpy array of shape (n, 3)
        """
        Tr = np.vstack((self.Tr_velo_to_cam, [0, 0, 0, 1]))
        homogenous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = np.dot(Tr, homogenous_points.T).T
        return points[:, :3]
    
    def project_one_point(self, point):
        """
        Project 3D point to 2D image
        :param point: numpy array of shape (3,)
        """
        point = np.dot(self.P0, np.hstack((point, 1)))      # P0, P1 is the projection matrix
        point = point / point[2]
        return point[:2]

    def project_points(self, points):
        """
        Project 3D points to 2D image
        :param points: numpy array of shape (n, 3)
        """
        points = np.dot(self.P0, np.hstack((points, np.ones((points.shape[0], 1)))).T)
        points = points / points[2]
        return points[:2].T



if __name__ == "__main__":
    yaml_path = "/home/sorodo/SLAM/dcp_Virconv/uitls/stf_cfg.yaml"
    stf = STFTools(yaml_path)
    print(stf.Tr_velo_to_cam)
    scan = stf.read_velodyne_bin("/mnt/h/data/seeingthroughfog/training/velodyne/2018-02-03_20-48-35_00500.bin")
    points = scan[:, :3]
    cam_points = stf.transform_points(points)
    print(cam_points[0])
    projected_points = stf.project_points(cam_points)
    print(projected_points[0])
    img0 = cv2.imread("/mnt/h/data/seeingthroughfog/training/image_2/2018-02-03_20-48-35_00500.png")
    img = np.zeros_like(img0)
    for point in projected_points:
        if point[0] < 0 or point[1] < 0:
            continue
        if point[0] > img0.shape[1] or point[1] > img0.shape[0]:
            continue
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
