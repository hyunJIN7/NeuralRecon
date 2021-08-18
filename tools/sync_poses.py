import os
import argparse

                        #이미지의 data.csv , gt data.csv
def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    """Load camera intrinsics"""  # frane.txt -> camera intrinsics
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f:  # frame.txt 읽어서
        cam_intrinsic_lines = f.readlines()

    cam_intrinsics = []
    for line in cam_intrinsic_lines[1:]:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_intrinsics.append([int(line_data_list[0])])  # timestamp,image.png 중에 timestamp만
                            #TODO: 기존에 float로 하면 부동소수점으로 표현으로 바뀌면서 1.xxe+18 이렇게됨 그래서 int로

    """load camera poses"""  # ARPose.txt -> camera pose
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()

    cam_poses = []
    for line in cam_pose_lines[1:]:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])

    # #timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
    # q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [],
    # v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1],
    # b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1],
    # b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]

    # ARposes.txt
    # time  tx(m) ty  tz
    # qw     qx   qy  qz
    """ outputfile로 syncpose 맞춰서 내보냄  """
    lines = []
    ip = 0
    length = len(cam_poses)
    # campose와 intrinsic 으로 뭘 계산해서 line에 넣음
    for i in range(len(cam_intrinsics)):  #
        while ip + 1 < length and abs(cam_poses[ip + 1][0] - cam_intrinsics[i][0]) < abs(
                cam_poses[ip][0] - cam_intrinsics[i][0]):
            ip += 1
        cam_pose = cam_poses[ip][:4] + cam_poses[ip][5:8] + [cam_poses[ip][4]]
        line = [str(a) for a in cam_pose]
        # line = [str(a) for a in cam_poses[ip]]
        line[0] = str(cam_intrinsics[i][0])
        lines.append(' '.join(line) + '\n')

    dirname = os.path.dirname(out_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_file, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_file', type=str, default='../Frames.txt')
    parser.add_argument('--pose_file', type=str, default='../ARPoses.txt')
    parser.add_argument('--out_file', type=str, default='../SyncedPoses.txt')
    args = parser.parse_args()
    sync_intrinsics_and_poses(args.cam_file, args.pose_file, args.out_file)
