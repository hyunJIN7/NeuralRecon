import os
import argparse

                            # cam_file에 rgb.txt
def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    #tum의 카메라 intrinsics 카메라 번호따라 잘 설정 ....TODO: check
    #TODO : sync만 맞추는거라 이게 필요 없을 것 같기도 , 그래서 일단 rgb파일로 맞춰
    cam_intrinsics_candi = [[0, 0, 517.3, 516.5, 318.6, 255.3], [0, 0, 520.9, 521.0, 325.1, 249.7], [0, 0, 535.4, 539.2, 320.1, 247.6]]
    cam_intrinsics = cam_intrinsics_candi[0]   #1차원으로 바꿈
    """Load camera intrinsics"""  # frane.txt -> camera intrinsics , 일단 RGB.txt 로드
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f: #rgb.txt : time filename     # frame.txt 읽어서  # time, image_num, fx,fy cx,cy  [[]]
        # cam_intrinsic_lines = f.readlines()
        rgb_timestamp_lines = f.readlines()

    #timestamp 비교 위한 rgb 파일
    rgb_timestamps = []
    for line in rgb_timestamp_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        rgb_timestamps.append([float(i) for i in line_data_list])
        # frame.txt -> cam_instrinsic


    """load camera poses"""
    # ARPose.txt    ->    time, tx, ty, tz, qw, qx, qy, qz
    # groundtruth.txt  -> time, tx, ty, tz, qx, qy, qz, qw
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()

    #TODO:timestamp 맞추는작업 필요
    cam_poses = []
    for line in cam_pose_lines[3:]:  #TODO: txt 파일 3줄 건너뛰어야해서 , 맞는지 확인
        line_data_list = line.split(' ')  # ' ' 으로 구분해야함
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])


    """ outputfile 로 syncedpose.txt 맞춰서 내보냄  """
    lines = []
    ip = 0
    length = len(cam_poses)
    #campose와 intrinsic 으로 뭘 계산해서 line에 넣음
    # cam_intrinsics[i][] ->  cam_intrinsics[]로 바꿈
    # TODO: cam_intrinsics 고정값이라 range(len(cam_intrinsics)) -> len_cam_poses 로 바꿔놨는데(어챂피 len 은 같을테니 ?..) 확인 필요
    for i in range(len(rgb_timestamps)):  # 의미하는바? 적당히 몇개 데이터만 뽑는건가
        while ip + 1 < length and abs(cam_poses[ip + 1][0] - cam_intrinsics[0]) < abs(cam_poses[ip][0] - cam_intrinsics[0]):
            ip += 1                    #    다음ip psoe의 time - 0번째랑 time 차  <   0번째랑 ip랑 time 차   이게 가능한가?? 다음꺼랑 차이가 더 커지는거 아닌가 아 원래는 i번째 여서 그런가
                                       # 차이가 덜 나는 것으로 찾는건가 ??????? 여기 어쩌냐
                                       #  rgb 파일에 timestamp와 비교해야할듯
        cam_pose = cam_poses[ip][:4] + cam_poses[ip][5:] + [cam_poses[ip][4]]
        line = [str(a) for a in cam_pose]
        # line = [str(a) for a in cam_poses[ip]]
        line[0] = str(i).zfill(5)
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
    