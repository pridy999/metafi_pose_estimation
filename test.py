import numpy as np
import csv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os.path
from models.mynetwork import posenet
import torch.nn as nn
from torch.autograd import Variable
from evaluation import compute_pck_pckh

file_path = '/data2/msc/zyj/wifi_vision/gait/keypoints_P1_P22_downsample3.pkl'
df = pd.read_pickle(file_path)
data = df.values
data_train,data_test_all = train_test_split(data, test_size=0.4, random_state=42)
data_val, data_test = train_test_split(data_test_all, test_size=0.5, random_state=41)
data_train= np.delete(data_train,(2,3), axis = 1)
data_val= np.delete(data_val,(2,3), axis = 1)
data_test= np.delete(data_test,(2,3), axis = 1)


class Mydata(Dataset):
    def __init__(self, opt):
        if opt == 'train':
            self.data = data_train
        if opt == 'val':
            self.data = data_val
        if opt == 'test':
            self.data = data_test
        else:
            print('wrong opt')

    def __getitem__(self, item):
        img_dir = os.path.join('/data2/msc/zyj/wifi_vision/gait',self.data[item][0])
        csi = scio.loadmat(os.path.join('/data2/msc/zyj/wifi_vision/gait', self.data[item][2]))
        csi_amp = csi['CSIamp']
        csi_amp[np.isinf(csi['CSIamp'])] = np.nan
        for i in range(10):
            temp_col = csi_amp[:,:,i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

        csi_amp = torch.tensor((csi_amp - np.min(csi_amp)) / (np.max(csi_amp) - np.min(csi_amp)))
        csi_pha = torch.tensor(
            (csi['CSIphase'] - np.min(csi['CSIphase'])) / (np.max(csi['CSIphase']) - np.min(csi['CSIphase'])))
        keypoint = torch.tensor(self.data[item][1])
        x = keypoint[:,0]
        y = keypoint[:,1]
        c = keypoint[:,2]
        keymatrix = np.zeros([3,17,17])
        for row in range(17):
            for column in range(17):
                if row == column:
                    keymatrix[:,row,column] = [x[row], y[row], c[row]]
                else:
                    keymatrix[:,row,column] = [x[row]-x[column], y[row]-y[column], c[row]*c[column]]

        keymatrix = torch.tensor(keymatrix)
        #csi_data = torch.stack([csi_amp, csi_pha], dim=0)
        csi_data = torch.unsqueeze(csi_amp,0)
        isnan = torch.isinf(csi_amp).any()
        if isnan == True:
            print(isnan)

        return {'csi_data': csi_data, 'csi_phase': csi_pha, 'csi_amplitude': csi_amp, 'keypoint': keypoint, 'keymatrix':keymatrix, 'img_dir': img_dir}

    def __len__(self):

        return len(self.data)


dataset_test = Mydata('test')

dataloader_train = DataLoader(dataset_test, batch_size = 1, shuffle=False)

metafi = torch.load('/data1/msc/zyj/WiSPPN-attention-2/weights/wisppn_P22-best.pkl')
metafi = metafi.cuda().eval()

metafi = metafi.cuda()
criterion_L2 = nn.MSELoss().cuda()


loss = 0
test_loss_iter = []
pck_50_iter = []
pck_40_iter = []
pck_30_iter = []
pck_20_iter = []
pck_10_iter = []
pck_5_iter = []
with torch.no_grad():
    for i, data in enumerate(dataset_test):
        csi_data = data['csi_data']
        img_dir = data['img_dir']
        label = data['keymatrix']  # 4,17,3
        keypoint = data['keypoint'].unsqueeze(0)
        csi_data = csi_data.cuda().unsqueeze(0)
        csi_data = csi_data.type(torch.cuda.FloatTensor)
        label = label.cuda().unsqueeze(0)
        xy_keypoint = data['keypoint'][:, 0:2].cuda()

        confidence = data['keypoint'][:, 2:3].cuda()

        pred_xy_keypoint,time_sum = metafi(csi_data)  # 4,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()

        print('time: %.3f' % time_sum)

        loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))


        test_loss_iter.append(loss.cpu().detach().numpy())

        pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1).unsqueeze(dim=0)
        xy_keypoint = torch.transpose(xy_keypoint, 0, 1).unsqueeze(dim=0)
        pred_xy_keypoint = pred_xy_keypoint.cpu()
        xy_keypoint = xy_keypoint.cpu()
        pck = compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5)

        pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint , 0.5))
        pck_40_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.4))
        pck_30_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.3))
        pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint , 0.2))
        pck_10_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.1))
        pck_5_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.05))

    test_mean_loss = np.mean(test_loss_iter)
    pck_50 = np.mean(pck_50_iter, 0)
    pck_40 = np.mean(pck_40_iter, 0)
    pck_30 = np.mean(pck_30_iter, 0)
    pck_20 = np.mean(pck_20_iter, 0)
    pck_10 = np.mean(pck_10_iter, 0)
    pck_5 = np.mean(pck_5_iter, 0)
    pck_50_overall = pck_50[17]
    pck_40_overall = pck_40[17]
    pck_30_overall = pck_30[17]
    pck_20_overall = pck_20[17]
    pck_10_overall = pck_10[17]
    pck_5_overall = pck_5[17]

    print('test result with loss: %.3f, pck_50: %.3f, pck_40: %.3f, pck_30: %.3f, pck_20: %.3f, pck_10: %.3f, pck_5: %.3f' % (test_mean_loss, pck_50_overall,pck_40_overall, pck_30_overall,pck_20_overall, pck_10_overall,pck_5_overall))
    print('-----pck_50-----')
    print(pck_50)
    print('-----pck_40-----')
    print(pck_40)
    print('-----pck_30-----')
    print(pck_30)
    print('-----pck_20-----')
    print(pck_20)
    print('-----pck_10-----')
    print(pck_10)
    print('-----pck_5-----')
    print(pck_5)










