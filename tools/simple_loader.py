import torch
import numpy as np
import os
import cv2


def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _

#dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth)
class ScanNetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "pose", str(id) + ".txt"), delimiter=' ')

        # Read depth image and camera pose
        depth_im = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id) + ".png"), -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "color", str(id) + ".jpg")),
                                   cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth_im, color_image

#dataset = ARKitDataset(n_imgs, scene, args.data_path, args.max_depth)
class ARKitDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth, id_list=None):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        cam_pose = np.loadtxt(os.path.join(self.data_path, self.scene, "poses", str(id).zfill(5) + ".txt"), delimiter=' ')
                                                                        #pose

        # Read depth image and camera pose
        # print("shape:::: ", cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id).zfill(5) + ".png"), -1).shape)
        depth_im = cv2.imread(os.path.join(self.data_path, self.scene, "depth", str(id).zfill(5) + ".png"), -1).astype(np.float32)
                                                               # -1: IMREAD_UNCHAGED, 이미찌 파일 Alpha chnnel까지 포함해 읽기
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > self.max_depth] = 0

        # Read RGB image
        color_image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, self.scene, "images", str(id).zfill(5) + ".jpg")),  # color , .jpg
                                   cv2.COLOR_BGR2RGB)
        #color_image = cv2.resize(color_image, (depth_im.shape[1], depth_im.shape[0]), interpolation=cv2.INTER_AREA)

        return cam_pose, depth_im, color_image
        #depth_im
