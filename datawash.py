import os
# import numpy as np
# import open3d as o3d

# lidar_path = "H:/data/seeingthroughfog/training/velodyne/2018-02-03_20-49-38_00000.bin"
# pointcloud = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

# # 提取x、y、z坐标
# xyz = pointcloud[:, :3]

# # 创建点云对象并可视化
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz)
# o3d.visualization.draw_geometries([pcd])

def copy_files(name_file='1.txt', content_file='2.txt'):
    """
    # 从指定的文件复制内容到多个新文件
    # 参数:
    #   name_file: 包含目标文件名列表的文件路径，默认为'1.txt'
    #   content_file: 包含要复制内容的文件路径，默认为'2.txt'
    """
    # 读取文件名列表
    with open(name_file, 'r', encoding='utf-8') as f:
        filenames = f.readlines()
    
    # 读取源文件内容
    with open(content_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 去除文件名中的换行符
    filenames = [name.strip() for name in filenames]
    
    # 为每个文件名创建对应的文件并写入内容
    for filename in filenames:
        # 确保文件名有.txt后缀
        if not filename.endswith('.txt'):
            filename = filename + '.txt'
            
        # content_file的目录
        target_dir = os.path.dirname(content_file)
        
        # 在content_file的目录下创建新文件并写入内容
        full_path = os.path.join(target_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'已创建文件: {full_path}')

file1 = 'H:/data/seeingthroughfog/ImageSets/train.txt'
file2 = 'H:/data/seeingthroughfog/training/calib/kitti_stereo_velodynehdl_calib.txt'

if __name__ == '__main__':
    copy_files(file1, file2)
