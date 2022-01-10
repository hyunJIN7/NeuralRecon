# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [NeuralRecon](https://github.com/zju3dv/NeuralRecon) by Yiming Xie and Jiaming Sun.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from PIL import Image, ImageOps
import numpy as np
from utils import coordinates
import transforms3d
import torch
from tools.tsdf_fusion.fusion import TSDFVolumeTorch

#추가
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/tranforms_epx1')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
# from util.camera_pose_visualizer import CameraPoseVisualizer


class Compose(object):
    """ Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToTensor(object):
    """ Convert to torch tensors"""

    def __call__(self, data):
        data['imgs'] = torch.Tensor(np.stack(data['imgs']).transpose([0, 3, 1, 2]))
        data['intrinsics'] = torch.Tensor(data['intrinsics'])
        data['extrinsics'] = torch.Tensor(data['extrinsics'])
        if 'depth' in data.keys():
            data['depth'] = torch.Tensor(np.stack(data['depth']))
        if 'tsdf_list_full' in data.keys(): # 전체 1로 차있는
            for i in range(len(data['tsdf_list_full'])):
                if not torch.is_tensor(data['tsdf_list_full'][i]):
                    data['tsdf_list_full'][i] = torch.Tensor(data['tsdf_list_full'][i])
        return data

#transforms.IntrinsicsPoseToProjection(cfg.TEST.N_VIEWS, 4)]   N_VIEWS 9
class IntrinsicsPoseToProjection(object):
    """ Convert intrinsics and extrinsics matrices to a single projection matrix
        World에서 Film coords로
    """

    def __init__(self, n_views, stride=1):
        self.nviews = n_views
        self.stride = stride

        #여기가 frame들 정렬 맞추는 곳
    def rotate_view_to_align_xyplane(self, Tr_camera_to_world): #extrinsic (9,4,4)
        # world space normal [0, 0, 1](xy plane)  camera space normal [0, -1, 0]
        z_c = np.dot(np.linalg.inv(Tr_camera_to_world), np.array([0, 0, 1, 0]))[: 3]  #xy plane을 world to camera 내적..의미가 뭘까
        axis = np.cross(z_c, np.array([0, -1, 0])) # 벡터 곱 연산,외적 , 수직일때 최대, 평행일때 최소    Inverse of a square matrix  https://rfriend.tistory.com/tag/%EA%B5%90%EC%B0%A8%EA%B3%B1
        axis = axis / np.linalg.norm(axis)  #사이즈 1로 맞춤
        theta = np.arccos(-z_c[1] / (np.linalg.norm(z_c))) #y값 z_c
        quat = transforms3d.quaternions.axangle2quat(axis, theta)   # Quaternion for rotation of angle `theta` around `vector`
        rotation_matrix = transforms3d.quaternions.quat2mat(quat)  #Calculate rotation matrix corresponding to quaternion
        return rotation_matrix      #quaternion : 방향,회전 표현, (x,y,z,w) 벡터(x,y,z)와 스칼라 (z)  , 짐벌락 문제 발생안함  https://hub1234.tistory.com/21

    def __call__(self, data):
        middle_pose = data['extrinsics'][self.nviews // 2]  # //:몫 연산자 ,  n_view의 중간 pose
        rotation_matrix = self.rotate_view_to_align_xyplane(middle_pose)
        rotation_matrix4x4 = np.eye(4)
        rotation_matrix4x4[:3, :3] = rotation_matrix
        data['world_to_aligned_camera'] = torch.from_numpy(rotation_matrix4x4).float() @ middle_pose.inverse()
        #world에 정렬된 camera

        proj_matrices = []
        for intrinsics, extrinsics in zip(data['intrinsics'], data['extrinsics']):
            view_proj_matrics = []
            for i in range(3):
                # from (camera to world) to (world to camera)
                proj_mat = torch.inverse(extrinsics.data.cpu())
                scale_intrinsics = intrinsics / self.stride / 2 ** i
                scale_intrinsics[-1, -1] = 1
                proj_mat[:3, :4] = scale_intrinsics @ proj_mat[:3, :4]
                view_proj_matrics.append(proj_mat)
            view_proj_matrics = torch.stack(view_proj_matrics)
            proj_matrices.append(view_proj_matrics)
        data['proj_matrices'] = torch.stack(proj_matrices)
        data.pop('intrinsics')
        data.pop('extrinsics')
        return data


def pad_scannet(img, intrinsics):
    """ Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    image 비율 4:3맞게 이미지 사이즈 늘려주고 이에 맞춰 instrinsc의 주점도 변화시킴.
    """
    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2)) # ( 1296, 968 ) -> ( 1296, 972 ) height
        intrinsics[1, 2] += 2 #이미지 크기 늘린만큼 그냥 늘려주나??...
    return img, intrinsics


class ResizeImage(object):  #(640,480) 으로 고정시킴
    """ Resize everything to given size.

    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for i, im in enumerate(data['imgs']):
            im, intrinsics = pad_scannet(im, data['intrinsics'][i]) #이미지 비율 4:3으로 맞추는 과정, Scannet 기준 세로 4pixel 늘리기
            #intrinsics (9,3,3)  이미지 하나당 9개? ..
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)  #bilinear interpolation으로 이미지  (640,480)  resize
            #카메라 내부 매개변수 알면 픽셀좌표와 정규 좌표 사이 변환 가능하니,이미지 -> (640,480)으로  resize 했으니 intrinsics도 그 비율에 맞춰 변화
            intrinsics[0, :] /= (w / self.size[0])  #640
            intrinsics[1, :] /= (h / self.size[1])

            data['imgs'][i] = np.array(im, dtype=np.float32)
            data['intrinsics'][i] = intrinsics

        return data

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomTransformSpace(object):
    """ Apply a random 3x4 linear transform to the world coordinate system.
        This affects pose as well as TSDFs.
    """

# transforms.RandomTransformSpace(
#                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation=False, random_translation=False,
#                  paddingXY=0, paddingZ=0, max_epoch=cfg.TRAIN.EPOCHS)
    def __init__(self, voxel_dim, voxel_size, random_rotation=True, random_translation=True,
                 paddingXY=1.5, paddingZ=.25, origin=[0, 0, 0], max_epoch=999, max_depth=3.0):
        """
        Args:
            voxel_dim: tuple of 3 ints (nx,ny,nz) specifying
                the size of the output volume 출력볼륨크기
            voxel_size: floats specifying the size of a voxel  0.04m , 최종 복셀 크기
            random_rotation: wheater or not to apply a random rotation
            random_translation: wheater or not to apply a random translation
            paddingXY: amount to allow croping beyond maximum extent of TSDF, max TSDF 값 넘어서면 crop할 양
            paddingZ: amount to allow croping beyond maximum extent of TSDF
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            max_epoch: maximum epoch
            max_depth: maximum depth
        """

        self.voxel_dim = voxel_dim
        self.origin = origin
        self.voxel_size = voxel_size
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.max_depth = max_depth
        self.padding_start = torch.Tensor([paddingXY, paddingXY, paddingZ])
        # no need to pad above (bias towards floor in volume)
        self.padding_end = torch.Tensor([paddingXY, paddingXY, 0])

        # each epoch has the same transformation   0~1 사이 값으로
        self.random_r = torch.rand(max_epoch)  #shape(epoch)
        self.random_t = torch.rand((max_epoch, 3))

    def __call__(self, data):
        origin = torch.Tensor(data['vol_origin'])
        if (not self.random_rotation) and (not self.random_translation):
            T = torch.eye(4) #대각만 1인 2차 텐서
        else:
            # construct rotaion matrix about z axis
            if self.random_rotation:
                r = self.random_r[data['epoch'][0]] * 2 * np.pi
            else:
                r = 0
            # first construct it in 2d so we can rotate bounding corners in the plane
            # 평면에서 bounding corner를 2d rotate, z축 기준 회전이니까
            R = torch.tensor([[np.cos(r), -np.sin(r)],
                              [np.sin(r), np.cos(r)]], dtype=torch.float32)

            # get corners of bounding volume
            # tsdf_list_full[0].shape  * 0.04  : 1로된 복셀 x,y,z 수에다 복셀 실제 크기 곱해 bounding volume 만들어놓기
            voxel_dim_old = torch.tensor(data['tsdf_list_full'][0].shape) * self.voxel_size #0.04
            xmin, ymin, zmin = origin #지금 scene의 origin(world 기준인듯) generate_gt 파일 확인 필요
            xmax, ymax, zmax = origin + voxel_dim_old
            #origin으로 부터 실제 사이즈의 bounding volume 범위 만들어 놓음

            corners2d = torch.tensor([[xmin, xmin, xmax, xmax],
                                      [ymin, ymax, ymin, ymax]], dtype=torch.float32)

            # rotate corners in plane
            corners2d = R @ corners2d

            # get new bounding volume (add padding for data augmentation)
            xmin = corners2d[0].min()
            xmax = corners2d[0].max()
            ymin = corners2d[1].min()
            ymax = corners2d[1].max()
            zmin = zmin
            zmax = zmax

            # randomly sample a crop
            voxel_dim = list(data['tsdf_list_full'][0].shape)
            start = torch.Tensor([xmin, ymin, zmin]) - self.padding_start
            end = (-torch.Tensor(voxel_dim) * self.voxel_size +
                   torch.Tensor([xmax, ymax, zmax]) + self.padding_end)  #end에서 voxel_dim은 왜 빼는거지???
            if self.random_translation:
                t = self.random_t[data['epoch'][0]]
            else:
                t = .5
            t = t * start + (1 - t) * end - origin

            T = torch.eye(4)

            T[:2, :2] = R
            T[:3, 3] = -t

        np.save('extrinsics_before', data['extrinsics'] )
        for i in range(len(data['extrinsics'])):
            data['extrinsics'][i] = T @ data['extrinsics'][i]

        data['vol_origin'] = torch.tensor(self.origin, dtype=torch.float, device=T.device)
        np.save('extrinsics_after', data['extrinsics'] )
        np.save('T_transforms_random',T)
        data = self.transform(data, T.inverse(), old_origin=origin)
        return data

    def transform(self, data, transform=None, old_origin=None,
                  align_corners=False):
        """ Applies a 3x4 linear transformation to the TSDF.

        Each voxel is moved according to the transformation and a new volume
        is constructed with the result.

        Args:
            data: items from data loader
            transform: 4x4 linear transform
            old_origin: origin of the voxel volume (xyz position of voxel (0, 0, 0))
                default (None) is the same as the input
            align_corners:

        Returns:
            Items with new TSDF and occupancy in the transformed coordinates
        """

        # ----------computing visual frustum hull------------
        bnds = torch.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf

        # 10개 view frustum 가지고  min,max boundary 구함
        for i in range(data['imgs'].shape[0]):  #n개의 이미지
            size = data['imgs'][i].shape[1:]
            cam_intr = data['intrinsics'][i]
            cam_pose = data['extrinsics'][i]
            view_frust_pts = get_view_frustum(self.max_depth, size, cam_intr, cam_pose) #max_dept = 3
            bnds[:, 0] = torch.min(bnds[:, 0], torch.min(view_frust_pts, dim=1)[0])
            bnds[:, 1] = torch.max(bnds[:, 1], torch.max(view_frust_pts, dim=1)[0])

        # -------adjust volume bounds-------
        num_layers = 3
        center = (torch.tensor(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2, -0.2)) - data['vol_origin']) / self.voxel_size
        center[:2] = torch.round(center[:2] / 2 ** num_layers) * 2 ** num_layers
        center[2] = torch.floor(center[2] / 2 ** num_layers) * 2 ** num_layers
        origin = torch.zeros_like(center)
        origin[:2] = center[:2] - torch.tensor(self.voxel_dim[:2]) // 2
        origin[2] = center[2]
        vol_origin_partial = origin * self.voxel_size + data['vol_origin']

        data['vol_origin_partial'] = vol_origin_partial

        # ------get partial tsdf and occupancy ground truth--------
        if 'tsdf_list_full' in data.keys():
            # -------------grid coordinates------------------
            old_origin = old_origin.view(1, 3)

            x, y, z = self.voxel_dim
            coords = coordinates(self.voxel_dim, device=old_origin.device)
            world = coords.type(torch.float) * self.voxel_size + vol_origin_partial.view(3, 1)
            world = torch.cat((world, torch.ones_like(world[:1])), dim=0)
            world = transform[:3, :] @ world
            coords = (world - old_origin.T) / self.voxel_size

            data['tsdf_list'] = []
            data['occ_list'] = []

            for l, tsdf_s in enumerate(data['tsdf_list_full']):
                # ------get partial tsdf and occ-------
                vol_dim_s = torch.tensor(self.voxel_dim) // 2 ** l
                tsdf_vol = TSDFVolumeTorch(vol_dim_s, vol_origin_partial,
                                           voxel_size=self.voxel_size * 2 ** l, margin=3)
                for i in range(data['imgs'].shape[0]):
                    depth_im = data['depth'][i]
                    cam_intr = data['intrinsics'][i]
                    cam_pose = data['extrinsics'][i]

                    # Integrate an RGB-D frame into the TSDF volume.
                    tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)

                tsdf_vol, weight_vol = tsdf_vol.get_volume()
                occ_vol = torch.zeros_like(tsdf_vol).bool()
                occ_vol[(tsdf_vol < 0.999) & (tsdf_vol > -0.999) & (weight_vol > 1)] = True

                # grid sample expects coords in [-1,1]
                coords_world_s = coords.view(3, x, y, z)[:, ::2 ** l, ::2 ** l, ::2 ** l] / 2 ** l
                dim_s = list(coords_world_s.shape[1:])
                coords_world_s = coords_world_s.view(3, -1)

                old_voxel_dim = list(tsdf_s.shape)

                coords_world_s = 2 * coords_world_s / (torch.Tensor(old_voxel_dim) - 1).view(3, 1) - 1
                coords_world_s = coords_world_s[[2, 1, 0]].T.view([1] + dim_s + [3])

                # bilinear interpolation near surface,
                # no interpolation along -1,1 boundry
                tsdf_vol = torch.nn.functional.grid_sample(
                    tsdf_s.view([1, 1] + old_voxel_dim),
                    coords_world_s, mode='nearest', align_corners=align_corners
                ).squeeze()
                tsdf_vol_bilin = torch.nn.functional.grid_sample(
                    tsdf_s.view([1, 1] + old_voxel_dim), coords_world_s, mode='bilinear',
                    align_corners=align_corners
                ).squeeze()
                mask = tsdf_vol.abs() < 1
                tsdf_vol[mask] = tsdf_vol_bilin[mask]

                # padding_mode='ones' does not exist for grid_sample so replace
                # elements that were on the boarder with 1.
                # voxels beyond full volume (prior to croping) should be marked as empty
                mask = (coords_world_s.abs() >= 1).squeeze(0).any(3)
                tsdf_vol[mask] = 1

                data['tsdf_list'].append(tsdf_vol)
                data['occ_list'].append(occ_vol)
            data.pop('tsdf_list_full')
            data.pop('depth')
        data.pop('epoch')
        return data

    def __repr__(self):
        return self.__class__.__name__


def rigid_transform(xyz, transform):
    """ Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = torch.cat([xyz, torch.ones((len(xyz), 1))], dim=1)
    xyz_t_h = (transform @ xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(max_depth, size, cam_intr, cam_pose):
    """ Get corners of 3D camera view frustum of depth image
    """
    im_h, im_w = size
    im_h = int(im_h)
    im_w = int(im_w)
    view_frust_pts = torch.stack([
        (torch.tensor([0, 0, 0, im_w, im_w]) - cam_intr[0, 2]) * torch.tensor( [0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[0, 0],
        (torch.tensor([0, 0, im_h, 0, im_h]) - cam_intr[1, 2]) * torch.tensor([0, max_depth, max_depth, max_depth, max_depth]) / cam_intr[1, 1],
        torch.tensor([0, max_depth, max_depth, max_depth, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T


    return view_frust_pts
