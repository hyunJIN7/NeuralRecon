import torch
from torch.nn.functional import grid_sample


def back_project(coords, origin, voxel_size, feats, KRcam):
    '''
    Unproject the image features to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix          3D camera coordi ->
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape  #TODO:feature shape 확인 필요,back project 어떻게 되는지  

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda() #voxel 수, feat의 c+1
    count = torch.zeros(coords.shape[0]).cuda() #voxel cnt 만큼 (13824,)

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1) #해당 배치인것에서 0아닌 값만 뽑아내고
        coords_batch = coords[batch_ind][:, 1:] # 해당 배치에  coor,x,y,z

        #batch 만큼 데이터 분리
        coords_batch = coords_batch.view(-1, 3) #(voxel수 , 3)
        origin_batch = origin[batch].unsqueeze(0)  # 그전voxel
        feats_batch = feats[:, batch]  #feature
        proj_batch = KRcam[:, batch]  #projection matrix

        grid_batch = coords_batch * voxel_size + origin_batch.float()  #(13824,3 )
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)  #(9,13284,3)으로 확장
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1) #(9,4,13284)

        # Project grid
        im_p = proj_batch @ rs_grid #(9,4,복셀수)=(9,4,4) @ (9,4,복셀 수)
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1) #*2 -1 : [-1,1]범위로 만드는.. rs_grid project 된 2D x,y 값
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w) #(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)       # (9,1,voxel 개수 ,2)   #flow field of shape
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True) #(n_views, c,h,w)
        # grid를 만들어

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()

        # aggregate multi view
        features = features.sum(dim=0) #nview features 를 합친다
        mask = mask.sum(dim=0)
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)
        features /= in_scope_mask
        features = features.permute(1, 0).contiguous()

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count
