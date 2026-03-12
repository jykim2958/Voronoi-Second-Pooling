# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad
# Warsaw University of Technology
import numpy as np
import torchvision.transforms as transforms

from datasets.augmentation import JitterPoints, RemoveRandomPoints, RandomTranslation, RemoveRandomBlock, RandomRotation, MaxClipRandomPoints
from datasets.base_datasets import TrainingDataset
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader, WildPlacesPointCloudLoader


class PNVTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = PNVPointCloudLoader()


class WildPlacesTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = WildPlacesPointCloudLoader()


class TrainTransform:
    # Augmentations specific for PointNetVLAD datasets (RobotCar and Inhouse)
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            self.transform = None
            return

        if self.aug_mode == 1:
            # Augmentations without random rotation around z-axis
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        # elif self.aug_mode == 2:
        #     t = [JitterPoints(sigma=0.01, clip=0.03), MaxClipRandomPoints(35000), RandomRotation(axis=np.array([0., 0., 1.]))]
        # elif self.aug_mode == 3:
        #     t = [MaxClipRandomPoints(35000), JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
        #          RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        # elif self.aug_mode == 4:
        #     t = [MaxClipRandomPoints(20000), JitterPoints(sigma=0.01, clip=0.03), RemoveRandomPoints(r=(0.0, 0.1)),
        #          RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        # elif self.aug_mode == 5:
        #     t = [FarthestPointSample(4096), JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
        #          RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        elif self.aug_mode == 2:
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4), RandomRotation(axis=np.array([0., 0., 1.]))]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

