import os
import glob
import warnings
import math

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from show_keypoints import imshow_keypoints
from mmcv.image import imwrite
import scipy.io as scio

#######################################
# 画图设置
skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
            [3, 5], [4, 6]]

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

pose_link_color = palette[[
    0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
]]
pose_kpt_color = palette[[
    16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
]]
kpt_score_thr = 0.3 # 阈值
radius = 7
thickness = 4

wisppn = torch.load('/data1/msc/zyj/WiSPPN-attention-2/weights/wisppn_P22-best-32-samenetwork-CTrans.pkl')
wisppn = wisppn.cuda().eval()
#wisppn = ResNet(ResidualBlock, [2, 2, 2, 2])
wisppn = wisppn.cuda()
criterion_L2 = nn.MSELoss().cuda()

###############################################
read_dir = '/data2/msc/zyj/wifi_vision/gait/MetaFi_processed_P22/gait_infra_png/P01_L1_A6/*.png'
for img_dir in glob.glob(read_dir):

    temp = img_dir.split('/')
    file = temp[8]
    name = temp[9]
    if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait', 'wifi', file, name)):
        dir = os.path.join('MetaFi_processed_P22/gait_infra_png',file,name)
        # img = mmcv.imread('/data2/msc/zyj/wifi_vision/gait/processed_data/gait_infra_png/P1_L1_A1/frame000127.png')
        img = mmcv.imread(img_dir)
        img = img.copy()
        img_h, img_w, _ = img.shape
        file_path = '/data2/msc/zyj/wifi_vision/gait/keypoints_P01_downsample3.pkl'
        df = pd.read_pickle(file_path)
        data = df.values
        m = np.where(data==dir)
        index = m[0]
        keypoint_ground_truth = data[index,1]
        keypoint_ground_truth=np.vstack(keypoint_ground_truth).astype(np.float)
        keypoint_ground_truth = torch.tensor(keypoint_ground_truth)
        csi_dir = ''.join(data[m[0],4])
        csi = scio.loadmat(os.path.join('/data2/msc/zyj/wifi_vision/gait', csi_dir))
        csi_amp = csi['CSIamp']
        csi_amp[np.isinf(csi['CSIamp'])] = np.nan
        for i in range(10): #32
            temp_col = csi_amp[:,:,i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        csi_amp = torch.tensor((csi_amp-np.min(csi_amp))/(np.max(csi_amp)-np.min(csi_amp))).unsqueeze(dim=0).unsqueeze(dim=0)
        csi_amp = csi_amp.type(torch.cuda.FloatTensor)
        pred_xy_keypoint,time = wisppn(csi_amp)
        pred_xy_keypoint = pred_xy_keypoint.squeeze()
        # m = torch.nn.AvgPool2d((1, 17))
        # pred_xy_keypoint = m(pred_xy).squeeze().unsqueeze(dim=0)
        #pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1)
        pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1).unsqueeze(dim=0)
        fake_score = torch.ones(1,1,17).cuda()
        pred_keypoint_add = torch.cat([pred_xy_keypoint , fake_score],dim=1).squeeze()
        pred_keypoint_add = torch.transpose(pred_keypoint_add, 0,1).cpu().detach().numpy()
        pred_keypoint_list = []
        pred_keypoint_list.append(pred_keypoint_add)
        xy_keypoint_list = []
        keypoint_ground_truth = keypoint_ground_truth.cpu().detach().numpy()
        xy_keypoint_list.append(keypoint_ground_truth)

        # save vision and wifi respectively
        if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait','wifi',file)):
            os.makedirs(os.path.join('/data2/msc/zyj/wifi_vision/gait','wifi',file))
        if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait','vision',file)):
            os.makedirs(os.path.join('/data2/msc/zyj/wifi_vision/gait','vision',file))

        img_dark = mmcv.imread('dark.png')
        img_dark = np.resize(img_dark,(480,640,3))
        imshow_keypoints(img_dark, pred_keypoint_list , skeleton, kpt_score_thr,
                        pose_kpt_color, pose_link_color, radius, thickness)
        imwrite(img_dark, os.path.join('/data2/msc/zyj/wifi_vision/gait','wifi',file,name))

        # imshow_keypoints(img, xy_keypoint_list , skeleton, kpt_score_thr,
        #                 pose_kpt_color, pose_link_color, radius, thickness)
        # imwrite(img, os.path.join('/data2/msc/zyj/wifi_vision/gait','vision',file,name))

        # #save vision and wifi at same image
        # if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait','vision_wifi_result',file)):
        #     os.makedirs(os.path.join('/data2/msc/zyj/wifi_vision/gait','vision_wifi_result',file))
        # imshow_keypoints(img, xy_keypoint_list, skeleton, kpt_score_thr,
        #                  pose_kpt_color, pose_link_color, radius, thickness)
        # imshow_keypoints(img, pred_keypoint_list, skeleton, kpt_score_thr,
        #                  pose_kpt_color, pose_link_color, radius, thickness)
        # imwrite(img, os.path.join('/data2/msc/zyj/wifi_vision/gait', 'vision_wifi_result', file, name))

        # # plot error at wifi image
        # if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait', 'wifi_error', file)):
        #     os.makedirs(os.path.join('/data2/msc/zyj/wifi_vision/gait','wifi_error',file))
        #
        # img_dark = mmcv.imread('dark.png')
        # img_dark = np.resize(img_dark,(480,640,3))
        # imshow_keypoints(img_dark, pred_keypoint_list , skeleton, kpt_score_thr,
        #                 pose_kpt_color, pose_link_color, radius, thickness)
        # for m in range(17):
        #     # plt.plot(pred_keypoint_add[m, 0:2], keypoint_ground_truth[m, 0:2], color='r')
        #     # plt.arrow(keypoint_ground_truth[m, 0], keypoint_ground_truth[m, 1], 0.3, -0.02, ec='red')
        #     a = pred_keypoint_add[m, 0:2].astype(int)
        #     b = keypoint_ground_truth[m, 0:2].astype(int)
        #     c = (100.1,200.1)
        #     d = (400.1, 600.1)
        #     cv2.arrowedLine(img_dark, a, b, (0, 0, 255), 3, 8, 0, 0.4)
        # imwrite(img_dark, os.path.join('/data2/msc/zyj/wifi_vision/gait', 'wifi_error', file, name))

        # # plot the error space
        # if not os.path.exists(os.path.join('/data2/msc/zyj/wifi_vision/gait', 'area_error_1', file)):
        #     os.makedirs(os.path.join('/data2/msc/zyj/wifi_vision/gait','area_error_1',file))
        # img_dark = mmcv.imread('dark.png')
        # img_dark = np.resize(img_dark,(480,640,3))
        #
        # # imshow_keypoints(img_dark, xy_keypoint_list, skeleton, kpt_score_thr,
        # #                  pose_kpt_color, pose_link_color, radius, thickness)
        # # combine_bone = [[5,7], [4,6], [6,7], [7,9], [9,11], [6,8], [8,10], [7,13], [6,12], [12,13], [13,15], [15,17], [12,14], [14,16]]
        # combine_bone = [[4, 6], [3, 5], [5, 6], [6, 8], [8, 10], [5, 7], [7, 9], [6, 12], [5, 11], [11, 12], [12, 14],
        #                 [14, 16], [11, 13], [13, 15]]
        # for m in range(len(combine_bone)):
        #     pt1 = combine_bone[m][0]
        #     pt2 = combine_bone[m][1]
        #     points = [pred_keypoint_add[pt1, 0:2].astype(int), pred_keypoint_add[pt2, 0:2].astype(int), keypoint_ground_truth[pt2, 0:2].astype(int), keypoint_ground_truth[pt1, 0:2].astype(int)]
        #     cv2.fillPoly(img_dark, [np.array(points)], color=[0,255,230])
        #     # cv2.fillPoly(img_dark, [np.array(points)], color=[0, 0, 255])
        # imshow_keypoints(img_dark, pred_keypoint_list, skeleton, kpt_score_thr,
        #                  pose_kpt_color, pose_link_color, radius, thickness)
        # imwrite(img_dark, os.path.join('/data2/msc/zyj/wifi_vision/gait', 'area_error_1', file, name))

