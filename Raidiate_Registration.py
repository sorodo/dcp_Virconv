path = 'E:/BaiduNetdiskDownload/radiate/fog_8_0'

import numpy as np
import matplotlib.pyplot as plt
import yaml,csv
import cv2
from uitls import calibration as calb, radiate as rad, depth as dp
import dark_channel as dc
from scipy.signal import convolve2d

# 将点云投影到图像上，并进行颜色映射
def project_lidar2img(img, lidar, lidar_extrinsics, cam_intrinsic, color_mode='same'):
    """
    Method to project the lidar into the camera

    :type img: np.array
    :param img: image with shape HxWx3

    :type lidar: np.array
    :param lidar: lidar point cloud with shape Nx5 (x,y,z,intensity,ring)

    :type lidar_extrinsics: np.array
    :param lidar_extrinsics: 4x4 matrix with lidar extrinsic parameters (Rotation
        and translations)

    :type cam_intrinsic: np.array
    :param cam_intrinsic: 3x3 matrix with camera intrinsic parameters in the form
        [[fx 0 cx],
        [0 fx cy],
        [0 0 1]]

    :type color_mode: string
    :param color_mode: what type of information is going to be representend in the lidar image
    options: 'same' always constant color. 'pseudo_distance': uses a color map to create a psedo
    color which refers to the distance. 'distance' creates an image with the actual distance as float

    :rtype: np.array
    :return: returns the projected lidar into the respective camera with the same size as the camera
    """
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    
    im_lidar = np.zeros_like(img)
    lidar_points = lidar[:, :3].T
    R = lidar_extrinsics[:3, :3]
    lidar_points = np.matmul(R, lidar_points).T
    lidar_points += lidar_extrinsics[:3, 3]
    for i in range(lidar.shape[0]):
        if (lidar_points[i, 2] > 0 and lidar_points[i, 2] < 80):
            xx = int(((lidar_points[i, 0] * fx) / lidar_points[i, 2]) + cx)
            yy = int(((lidar_points[i, 1] * fy) / lidar_points[i, 2]) + cy)
            if (xx > 0 and xx < img.shape[1] and yy > 0 and yy < img.shape[0]):
                if color_mode == 'same':
                    im_lidar = cv2.circle(
                        im_lidar, (xx, yy), 1, color=(0, 255, 0))
                elif color_mode == 'pseudo_distance':
                    dist = np.sqrt(lidar_points[i, 0] * lidar_points[i, 0] +
                                   lidar_points[i, 1] * lidar_points[i, 1] +
                                   lidar_points[i, 2] * lidar_points[i, 2])
                    norm_dist = np.array(
                        [(dist / 20) * 255]).astype(np.uint8)
                    cc = np.array(plt.get_cmap('viridis')(norm_dist)) * 255
                    im_lidar = cv2.circle(
                        im_lidar, (xx, yy), 1, color=cc[0][:3])
                elif color_mode == 'distance':
                    dist = np.sqrt(lidar_points[i, 0] * lidar_points[i, 0] +
                                   lidar_points[i, 1] * lidar_points[i, 1] +
                                   lidar_points[i, 2] * lidar_points[i, 2])
                    im_lidar[yy, xx] = dist

    return im_lidar


def camera_2d_calibration(img, cam_mat, k1, k2, p1, p2):
    """
    对相机图像进行二维校正。

    :type img: np.array
    :param img: 相机图像

    :type cam_mat: np.array
    :param cam_mat: 相机内参矩阵

    :type k1: float
    :param k1: 相机畸变参数k1

    :type k2: float
    :param k2: 相机畸变参数k2

    :type p1: float
    :param p1: 相机畸变参数p1

    :type p2: float
    :param p2: 相机畸变参数p2

    :rtype: np.array
    :return: 返回校正之后的相机图像
    """
    h, w = img.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(cam_mat, np.array([k1, k2, p1, p2, 0]), None, None, (w,h), cv2.CV_32FC1)
    dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return dst


def merge_images(background, overlay):
    """
    合并两个图像，一个作为背景，另一个图像黑色部分去除，其余部分覆盖到背景图上。

    :type background: np.array
    :param background: 背景图像

    :type overlay: np.array
    :param overlay: 要覆盖到背景上的图像

    :rtype: np.array
    :return: 返回合并后的图像
    """
    # 创建一个与背景图像相同的图像
    merged_image = background.copy()

    # 找到覆盖图像中非黑色的部分
    overlay_indices = np.where(overlay != 0)

    # 将覆盖图像的非黑色部分覆盖到背景图像上
    merged_image[overlay_indices] = overlay[overlay_indices]

    return merged_image


def project_to_camera_image(point_cloud, T, K):
    point_cloud_1 = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))
    point_cloud_1 = np.dot(T, point_cloud_1.T).T
    point_xyz = point_cloud_1[:, :3]
    point_distance = np.linalg.norm(point_xyz, axis=1)

    # 将3D点云投影到相机图像上
    point_xyz = np.dot(K, point_xyz.T).T
    point_xyz = point_xyz / point_xyz[:, 2].reshape(-1, 1)

    return np.column_stack((point_xyz[:,:2].reshape(-1, 2), point_distance))


# 读取CSV文件中的点云数据
def read_point_cloud_from_csv(file_path):
    """
    读取CSV文件中的点云数据。

    参数：
        file_path (str): CSV文件的路径。

    返回：
        tuple: 包含坐标和强度数据的元组 (xyz, intensity)。
    """
    xyz = []
    intensity = []
    channel = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                i = float(row[3])
                c = int(row[4])
                xyz.append([x, y, z])
                intensity.append(i)
                channel.append(c)

    return np.array(xyz), np.array(intensity), np.array(channel)


# 计算点云的深度，得到点云的深度图
def calculate_depth(point_cloud, T, K, img_size):

    point_cloud_1 = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))
    point_cloud_1 = np.matmul(T, point_cloud_1.T).T
    point_xyz = point_cloud_1[:, :3]
    point_distance = np.linalg.norm(point_xyz, axis=1)
    mask = (point_xyz[:, 2] >= 0) & (point_xyz[:, 2] <= 80)
    point_cloud_1 = point_cloud_1[mask]
    point_xyz = point_xyz[mask]
    point_distance = point_distance[mask]

    # 将3D点云投影到相机图像上
    projected_points = np.dot(K, point_xyz.T).T
    projected_points = projected_points / projected_points[:, 2].reshape(-1, 1)
    projected_points = projected_points[:, :2]
    
    valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 1] < img_size[0]) & (projected_points[:, 1] >= 0) & (projected_points[:, 0] < img_size[1])
    valid_points_distance = point_distance[valid_points]
    
    # 根据深度进行排序
    sorted_indices = np.argsort(valid_points_distance)[::-1]
    sorted_points_distance = valid_points_distance[sorted_indices]
    
    # 根据深度进行投影
    depth_image = np.zeros(img_size)
    for i, distance in enumerate(sorted_points_distance):
        x, y = projected_points[valid_points][sorted_indices[i]]
        x, y = int(x), int(y)
        
        
        # 把明度信息投影到x,y上面
        brightness = int(255 * (distance - sorted_points_distance.min()) / (sorted_points_distance.max() - sorted_points_distance.min()))
        # depth_image[x, y] = brightness
        brightness = plt.get_cmap('Greys')(brightness)
        
        # 在x,y周围的kernel_size大小范围内，修改为同一个明度brightness
        kernel_size = int(4 + (8 - 4) * (distance - sorted_points_distance.min()) / (sorted_points_distance.max() - sorted_points_distance.min()))
        kernel_size = max(4, min(kernel_size, 8))
        cv2.circle(depth_image, (x, y), kernel_size, color=brightness, thickness=-1)
        
    return depth_image


# 利用精确的深度信息，结合暗通道估计的透射率，修正透射率
def revise_transmission(dark_channel, point_cloud, T, K, img_size):
    point_cloud_1 = np.hstack((point_cloud, np.ones((point_cloud.shape[0],1))))
    point_cloud_1 = np.matmul(T, point_cloud_1.T).T
    point_xyz = point_cloud_1[:, :3]
    point_distance = np.linalg.norm(point_xyz, axis=1)
    mask = (point_xyz[:, 2] >= 0) & (point_xyz[:, 2] <= 80)
    point_cloud_1 = point_cloud_1[mask]
    point_xyz = point_xyz[mask]
    point_distance = point_distance[mask]

    # 将3D点云投影到相机图像上
    projected_points = np.dot(K, point_xyz.T).T
    projected_points = projected_points / projected_points[:, 2].reshape(-1, 1)
    projected_points = projected_points[:, :2]
    
    valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 1] < img_size[0]) & (projected_points[:, 1] >= 0) & (projected_points[:, 0] < img_size[1])
    valid_points_distance = point_distance[valid_points]
    
    # 根据深度进行排序
    sorted_indices = np.argsort(valid_points_distance)[::-1]
    sorted_points_distance = valid_points_distance[sorted_indices]

    beta_list = []

    transmission_image = dark_channel.copy()    
    # transmission_image = np.zeros_like(dark_channel)
    for i, distance in enumerate(sorted_points_distance):
        x, y = projected_points[valid_points][sorted_indices[i]]
        x, y = int(x), int(y)
        transmission = dark_channel[y, x]
        beta = -np.log(transmission) / distance
        beta_list.append(beta)
        if len(beta_list) > 100:
            beta_list.pop(0)
        beta = np.mean(beta_list)
        # print(beta)
        beta = 0.015
        # 把明度信息投影到x,y上面
        brightness = np.exp(-beta * distance)
        
        # 在x,y周围的kernel_size大小范围内，修改为同一个明度brightness
        kernel_size = int(5 + (10 - 5) * (distance - sorted_points_distance.min()) / (sorted_points_distance.max() - sorted_points_distance.min()))
        kernel_size = max(5, min(kernel_size, 10))
        cv2.circle(transmission_image, (x, y), kernel_size, color=brightness, thickness=-1)

    return transmission_image
   




# 示例用法
if __name__ == "__main__":
    file_path = 'D:\development\SLAM\Dehaze\pythonProject\data\default-calib.yaml'
    data_path = 'E:\BaiduNetdiskDownload/radiate/fog_8_0/velo_lidar/000076.csv'
    img_path = 'E:\BaiduNetdiskDownload/radiate/fog_8_0/zed_left/000064.png'
    right_img_path = 'E:\BaiduNetdiskDownload/radiate/fog_8_0/zed_right/000062.png'

    lidar_points, intensity, channel = read_point_cloud_from_csv(data_path)


    # 相机内参
    with open(file_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    repost = calb.Calibration(calib_data)

    radi = rad.Sequence('E:\BaiduNetdiskDownload/radiate/fog_8_0', 'D:\development\SLAM\Dehaze\pythonProject\data\config.yaml')

    # dst = radi.project_lidar(np.column_stack([lidar_points, intensity, channel]), repost.LidarToLeft, repost.left_cam_mat, color_mode='same')
    fst = cv2.imread(img_path)
    r_fst = cv2.imread(right_img_path)
    gray_fst = cv2.cvtColor(fst, cv2.COLOR_BGR2GRAY)

    point_cloud = dp.PointCloud(data_path)
    point_cloud.transform(repost.LidarToLeft)
    # 点云投影
    point_project = point_cloud.project(repost.left_cam_mat, fst.shape)
    # point_cloud.display()
    point_depth = dp.PointDepth(point_cloud, fst, r_fst)
    # 生成扫描线图像
    scanline_image = point_depth.generate_scanline_image(gray_fst.shape)
    cv2.imshow('scanline_image', scanline_image)
    cv2.waitKey(0)

    # cleaned_image = point_depth.remove_outliers(scanline_image)
    # cv2.imshow('cleaned_image', cleaned_image)
    # cv2.waitKey(0)
    #
    # interpolated_image = point_depth.interpolate_maxpool(cleaned_image)
    # cv2.imshow('interpolated_image', interpolated_image)
    # cv2.waitKey(0)

    # # 生成视差图
    # depth_image = point_depth.PMS(scanline_image, scanline_image, repost.left_cam_mat, repost.right_cam_mat,
    #                               repost.LeftT, repost.RightT)
    # cv2.imshow('depth_image', depth_image)
    # cv2.waitKey(0)
    # depth_image = point_depth.SGBM(fst, r_fst, repost.left_cam_mat, repost.right_cam_mat,
    #                               repost.LeftT, repost.RightT)
    # cv2.imshow('depth_image', depth_image)
    # cv2.waitKey(0)

    # dst = project_lidar2img(fst, np.column_stack([lidar_points, intensity, channel]), repost.LidarToLeft, repost.left_cam_mat, color_mode='pseudo_distance')
    # fst = camera_2d_calibration(fst, repost.left_cam_mat, repost.left_cam_dist[0], repost.left_cam_dist[1], repost.left_cam_dist[2], repost.left_cam_dist[3])

    # dst = merge_images(fst,dst)

    # 转换点云
    # world_points = project_to_camera_image(lidar_points, repost.LidarToLeft, repost.left_cam_mat)

    # dst = extract_colors_from_image(world_points, cv2.imread('E:/BaiduNetdiskDownload/radiate/fog_8_0/zed_left/001784.png'))


    # depth_image = calculate_depth(lidar_points, repost.LidarToLeft, repost.left_cam_mat, dst.shape)

    # 计算暗通道
    I = fst.astype('float64') / 255
    dark_channel = dc.DarkChannel(I, 3)

    # 计算大气光
    A = dc.AtmLight(I, dark_channel)

    # 计算透射率
    te = dc.TransmissionEstimate(I, A, 15)
    cv2.imshow('te', te)
    cv2.waitKey(0)

    # 修正透射率
    beta, p = dc.estimate_beta(te, scanline_image)
    transmission_image = dc.lidar_guided_filter(te, p, 15, 0.01)
    cv2.imshow('transmission_image', transmission_image)
    cv2.waitKey(0)


    t = dc.TransmissionRefine(fst, transmission_image)
    J = dc.Recover(I, t, A, 0.1)
    cv2.imshow('J', J)
    cv2.waitKey(0)

