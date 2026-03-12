import numpy as np
import os
import open3d as o3d

from datasets.base_datasets import PointCloudLoader
from pyntcloud import PyntCloud

class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        pc = np.fromfile(file_path, dtype=np.float64)
        pc = np.float32(pc)
        # coords are within -1..1 range in each dimension
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        return pc


class WildPlacesPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False # True
        self.remove_ground_plane = False # True # NOTE Don't do this for the segmented
        self.ground_plane_level = 0.5
        # self.remove_zero_points = False
        # self.remove_ground_plane = False
        # self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        pointcloud = PyntCloud.from_file(file_pathname)
        pc = np.array(pointcloud.points)[:,:3].astype(np.float32)
        return pc 

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        pc = np.ascontiguousarray(pc)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            pc = o3d.t.geometry.PointCloud(pc)
            _, inliers = pc.segment_plane(distance_threshold=self.ground_plane_level,
                                          ransac_n=3,
                                          num_iterations=100)
            pc = pc.select_by_index(inliers, invert=True)
            pc = np.asarray(pc.to_legacy().points).astype(np.float32)

        # # Downsample # NOTE don't do this for the segmented
        # # if len(pc) > 4096:
        # #     pc = o3d.t.geometry.PointCloud(pc)
        # #     pc = pc.farthest_point_down_sample(4096)
        # #     pc = np.asarray(pc.to_legacy().points).astype(np.float32)
        # pc = o3d.t.geometry.PointCloud(pc)
        # pc = pc.voxel_down_sample(voxel_size=0.1) # 0.1 0.5
        # pc = np.asarray(pc.to_legacy().points).astype(np.float32)


        # NOTE Variable Hieght Removal !!!
        # pc = pc[pc[:, 2] <= 6]

        # NOTE Size normalization
        pc = pc / 30 # a max diameter should be around 60m # NOTE 60 30
        return pc


