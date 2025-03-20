import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt


class PointCloud:
    def __init__(self, file_path):
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

        self.xyz = np.array(xyz)
        self.distance = np.linalg.norm(self.xyz, axis=1)
        self.intensity = np.array(intensity)
        self.channel = np.array(channel)
        self.project_xy = None


    def transform(self, lidar2cameraT):
        """
        将点云数据从激光雷达的坐标系变换到相机坐标系。

        参数：
        - transformation_matrix: 变换矩阵。

        返回：
        - transformed_points: 变换后的点云数据。
        """
        # 将点云转换为齐次坐标
        homogeneous_points = np.hstack([self.xyz, np.ones((len(self.xyz), 1))])
        # 使用变换矩阵进行变换
        transformed_points = np.dot(lidar2cameraT, homogeneous_points.T).T
        # 将变换后的点云转换为非齐次坐标
        transformed_points = transformed_points[:, :3]

        # 滤除在相机视角之外的点
        mask = (transformed_points[:, 2] >= 0) & (transformed_points[:, 2] <= 120)

        self.xyz = transformed_points[mask]
        self.intensity = self.intensity[mask]
        self.channel = self.channel[mask]
        self.distance = np.linalg.norm(self.xyz, axis=1)

        return transformed_points



    def project(self, camera_matrix, image_shape):
        """
        将点云投影到相机坐标系。

        参数：
        - camera_matrix: 相机内参矩阵。

        返回：
        - projected_points: 投影后的点云数据(N, (x,y,depth))。
        """
        # 3D 点投影到 2D 平面
        projected_points = np.dot(camera_matrix, self.xyz.T).T
        projected_points = projected_points / (self.xyz[:, 2].reshape(-1, 1))
        projected_points = projected_points[:, :2]
        projected_points = np.round(projected_points).astype(int)

        # 滤除在相机视角之外的点
        mask = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_shape[1]) & \
               (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_shape[0])
        projected_points = projected_points[mask]
        
        self.xyz = self.xyz[mask]
        self.distance = self.distance[mask]
        self.intensity = self.intensity[mask]
        self.channel = self.channel[mask]

        # 添加深度信息
        brightness = (self.distance / np.max(self.distance)).reshape(-1, 1)
        projected_points = np.hstack([projected_points, brightness])
        self.project_xy = projected_points

        return projected_points


    def display(self):
        """
        显示点云数据。
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2], c=self.intensity, cmap='gray', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()



class PointDepth:
    def __init__(self, pointcloud: PointCloud, left_cam_img, right_cam_img):
        """
        初始化 PointDepth 类。
        :param pointcloud: (N, 3) 的点云数据，包含 [x, y, depth]。
        :param left_cam_img:
        :param right_cam_img:
        """
        self.points = pointcloud.project_xy
        self.y = self.points[:, 1]
        self.unique_y = np.unique(self.y).astype(int)
        self.depth = pointcloud.xyz[:, 2]
        self.intensity = pointcloud.intensity
        self.channel = pointcloud.channel
        self.left_cam_img = left_cam_img
        self.right_cam_img = right_cam_img


    def generate_scanline_image(self, image_shape, x_interval=8):
        """
        将点云投影数据转换为扫描线图像。

        参数：
        - points: (N, 3) 的点云投影数据，包含 [x, y, value]，分别为横坐标、纵坐标和深度值/颜色值。
        - image_shape: 输出图像的形状 (height, width)。

        返回：
        - scanline_image: 生成的扫描线图像。
        """
        # 初始化图像
        scanline_image = np.zeros(image_shape)

        for y in self.unique_y:
            # 获取当前行的所有点
            row_points = self.points[self.points[:, 1] == y]

            # 按 x 坐标排序
            row_points = row_points[np.argsort(row_points[:, 0])]

            # 提取 x 和 value 值
            x_coords = row_points[:, 0].astype(int)
            values = row_points[:, 2]

            # 对 x 进行线性插值
            for i in range(len(x_coords) - 1):
                x1, x2 = x_coords[i], x_coords[i + 1]
                v1, v2 = values[i], values[i + 1]
                if x2 - x1 <= x_interval and x2 != x1:
                    for x in range(x1, x2 + 1):
                        v = (v1 + (v2 - v1) * (x - x1) / (x2 - x1)).astype(np.float32)
                        scanline_image[int(y), int(x)] = 1-v

        return scanline_image

    # 清除扫描线图像中的异常点
    def remove_outliers(self, scanline_image, line_threshold=11, depth_threshold=1e-3):
        """
        清除扫描线图像中的异常点。

        参数：
        - scanline_image: 扫描线图像。
        - threshold: 异常点阈值。

        返回：
        - cleaned_image: 清除异常点后的图像。
        """
        # 初始化清洗后的图像
        cleaned_image = np.copy(scanline_image)

        # 构造特殊卷积核
        kernel_size = line_threshold
        half_kernel = kernel_size // 2

        # 使用cv2库对图像周围补黑
        padded_image = cv2.copyMakeBorder(scanline_image, half_kernel, half_kernel, half_kernel, half_kernel, cv2.BORDER_CONSTANT, value=0)
        scanline_image = padded_image

        # 遍历每一行扫描线
        for y in self.unique_y:

            # 使用卷积核操作清除异常点
            for x in range(cleaned_image.shape[1]):
                center_depth = scanline_image[int(y)+half_kernel, int(x)+half_kernel]
                kernel = scanline_image[int(y):int(y) + 2*half_kernel + 1, int(x) + half_kernel]
                if np.any(kernel > center_depth + depth_threshold):
                    cleaned_image[int(y), int(x)] = 0

        return cleaned_image
    

    # 最大池化插值
    def interpolate_maxpool(self, scanline_image, x_interval=4):
        """
        对扫描线图像进行补全，采用最近邻插值的方法在两个有值的像素之间插值。

        参数：
        - scanline_image: 扫描线图像。

        返回：
        - interpolated_image: 纵向坐标补全后的图像。
        """
        # 初始化纵向坐标补全后的图像
        interpolated_image = np.copy(scanline_image)
        # 使用卷积核加速最近邻插值
        kernel_size = 2 * x_interval + 1
        padded_image = cv2.copyMakeBorder(scanline_image, x_interval, x_interval, x_interval, x_interval, cv2.BORDER_CONSTANT, value=0)

        for y in range(scanline_image.shape[0]):
            for x in range(scanline_image.shape[1]):
                if scanline_image[y, x].any() == 0:
                    kernel = padded_image[y:y + kernel_size, x:x + kernel_size]
                    non_zero_elements = kernel[kernel != 0]
                    if non_zero_elements.size > 0:
                        max_value = np.max(non_zero_elements)
                        interpolated_image[y, x] = max_value
        return interpolated_image
    

    # 导向滤波
    def guided_filter(self, scanline_image, camera_image):
        """
        对扫描线图像进行补全，采用导向滤波的方法在两个有值的像素之间插值。
        """
        pass


class DarkChanLidar:
    def __init__(self, project_lidar, image):
        self.project_lidar = project_lidar
        self.image = image
        self.dark_channel_img = self.dark_channel()

    
    def dark_channel(self):
        m = self.deHaze(self.image / 255.0) * 255
        cv2.imshow('defog', m.astype(np.uint8))
        cv2.waitKey(0)
        pass


    def zmMinFilterGray(self, src, r=7):
        '''''最小值滤波，r是滤波器半径'''
        return cv2.erode(src, np.ones((2 * r - 1, 2 * r - 1)))


    def guidedfilter(self, I, p, r, eps):
        '''''普通的引导滤波'''
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p

        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b
    

    def lidar_guided_filter(self, I, p, r, eps=1e-3):
        '''
        激光雷达的引导滤波
        输入：
        - I: 输入大气透过率估计图
        - p: 点云投影引导图像
        - r: 滤波器半径
        - eps: 正则化参数
        '''
        # 定义mask
        mask = p > 0

        # 定义窗口图f_k
        f_k = cv2.boxFilter(mask.astype(np.float32), -1, (r, r), normalize=False)

        # 计算m_p
        m_p = cv2.boxFilter(p + 1e-6, -1, (r, r), normalize=False) / f_k

        # 计算m_Ip
        m_Ip = cv2.boxFilter(I * p + 1e-6, -1, (r, r), normalize=False) / f_k

        # 计算m_II
        I_mask = I * mask
        m_II = cv2.boxFilter(I_mask * I_mask + 1e-6, -1, (r, r), normalize=False) / f_k

        # 其余步骤不变
        cov_Ip = m_Ip - m_p * m_p
        var_I = m_II - m_p * m_p

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_p

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b


    def getV1(self, m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
        '''''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = np.min(m, 2)  # 得到暗通道图像
        V1 = self.lidar_guided_filter(V1, self.project_lidar, r, eps)  # 使用引导滤波优化
        V1 = self.guidedfilter(V1, self.zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
        bins = 2000
        ht = np.histogram(V1, bins)  # 计算大气光照A
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

        V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

        return V1, A


    def deHaze(self, m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        Y = np.zeros(m.shape)
        V1, A = self.getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
        for k in range(3):
            Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
        Y = np.clip(Y, 0, 1)
        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
        return Y

