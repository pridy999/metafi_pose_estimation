import numpy as np
import csv
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os.path
from models.mynetwork import posenet, weights_init
import torch.nn as nn
from torch.autograd import Variable
from evaluation import compute_pck_pckh




file_path = '/data2/msc/zyj/wifi_vision/gait/keypoints_P1_P22_downsample3.pkl'
df = pd.read_pickle(file_path)
data = df.values
data_train,data_test_all = train_test_split(data, test_size=0.4, random_state=42)
data_val, data_test = train_test_split(data_test_all, test_size=0.5, random_state=41)
data_train= np.delete(data_train,(0,2,3), axis = 1)
data_val= np.delete(data_val,(0,2,3), axis = 1)
data_test= np.delete(data_test,(0,2,3), axis = 1)

#csi = scio.loadmat(os.path.join('/data2/msc/zyj/wifi_vision/gait', data_train[0][1]))


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
        csi = scio.loadmat(os.path.join('/data2/msc/zyj/wifi_vision/gait', self.data[item][1]))
        csi_amp = csi['CSIamp']
        csi_amp[np.isinf(csi['CSIamp'])] = np.nan
        for i in range(10): #32
            temp_col = csi_amp[:,:,i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()

        csi_amp = torch.tensor((csi_amp-np.min(csi_amp))/(np.max(csi_amp)-np.min(csi_amp)))
        csi_pha = torch.tensor((csi['CSIphase']-np.min(csi['CSIphase']))/(np.max(csi['CSIphase'])-np.min(csi['CSIphase'])))
        #csi_amp = torch.tensor(csi['CSIamp'])
        #csi_pha = torch.tensor(csi['CSIphase'])
        keypoint = torch.tensor(self.data[item][0])
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

        return {'csi_data': csi_data, 'csi_phase': csi_pha, 'csi_amplitude': csi_amp, 'keypoint': keypoint, 'keymatrix':keymatrix}

    def __len__(self):

        return len(self.data)



dataset_train = Mydata('train')
batchsize = 32
dataloader_train = DataLoader(dataset_train, batch_size = batchsize, shuffle=False)
dataset_val = Mydata('val')
dataloader_val = DataLoader(dataset_val, batch_size = 1, shuffle=False)
metafi = posenet()
metafi.apply(weights_init)
metafi = metafi.cuda()
criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.SGD(metafi.parameters(), lr = 0.001, momentum=0.9)
n_epochs = 20
n_epochs_decay = 30
epoch_count = 1
def lambda_rule(epoch):

    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1))


num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pck_50_overall_max = 0
train_mean_loss_iter = []
for epoch_index in range(num_epochs):

    loss = 0
    train_loss_iter = []
    metafi.train()

    for idx, data in enumerate(dataloader_train):

        csi_data = data['csi_data']
        label = data['keymatrix']#4,17,3
        csi_data = csi_data.cuda()
        csi_data = csi_data.type(torch.cuda.FloatTensor)

        label = label.cuda()
        mm=data['keypoint']
        xy_keypoint = data['keypoint'][:,:,0:2].cuda()
        confidence = data['keypoint'][:, :, 2:3].cuda()

        pred_xy_keypoint, time = metafi(csi_data) #b,2,17,17
        pred_xy_keypoint = pred_xy_keypoint.squeeze()

        loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))/batchsize
        train_loss_iter.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        lr = np.array(scheduler.get_last_lr())
        message = '(epoch: %d, iters: %d, lr: %.5f, loss: %.3f) ' % (epoch_index, idx * batchsize, lr, loss)
        print(message)
    scheduler.step()
    train_mean_loss = np.mean(train_loss_iter)
    train_mean_loss_iter.append(train_mean_loss)
    print('end of the epoch: %d, with loss: %.3f' % (epoch_index, train_mean_loss,))
    metafi.eval()
    valid_loss_iter = []
    pck_50_iter = []
    pck_20_iter = []
    with torch.no_grad():
        for i, data in enumerate(dataset_val):
            csi_data = data['csi_data']
            label = data['keymatrix']  # 4,17,3
            # xy_keypoint = data['keypoint'].unsqueeze(0)
            csi_data = csi_data.cuda().unsqueeze(0)
            csi_data = csi_data.type(torch.cuda.FloatTensor)
            label = label.cuda().unsqueeze(0)
            xy = label[:, 0:2, :, :]
            # confidence = label[:, 2:3, :, :]
            xy_keypoint = data['keypoint'][:, 0:2].cuda()
            confidence = data['keypoint'][:, 2:3].cuda()

            pred_xy_keypoint,time = metafi(csi_data)  # 4,2,17,17
            pred_xy_keypoint = pred_xy_keypoint.squeeze()

            loss = criterion_L2(torch.mul(confidence, pred_xy_keypoint), torch.mul(confidence, xy_keypoint))

            valid_loss_iter.append(loss.cpu().detach().numpy())

            pred_xy_keypoint = torch.transpose(pred_xy_keypoint, 0, 1).unsqueeze(dim=0)
            xy_keypoint = torch.transpose(xy_keypoint, 0, 1).unsqueeze(dim=0)
            pred_xy_keypoint = pred_xy_keypoint.cpu()
            xy_keypoint = xy_keypoint.cpu()
            pck = compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5)

            pck_50_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.5))
            pck_20_iter.append(compute_pck_pckh(pred_xy_keypoint, xy_keypoint, 0.2))

        valid_mean_loss = np.mean(valid_loss_iter)
        pck_50 = np.mean(pck_50_iter,0)
        pck_20 = np.mean(pck_20_iter,0)
        pck_50_overall = pck_50[17]
        pck_20_overall = pck_20[17]
        print('validation result with loss: %.3f, pck_50: %.3f, pck_20: %.3f' % (valid_mean_loss, pck_50_overall,pck_20_overall))

        if pck_50_overall > pck_50_overall_max:
            print('saving the model at the end of epoch %d with pck_50: %.3f' % (epoch_index, pck_50_overall))
            torch.save(metafi, '/data1/msc/zyj/WiSPPN-attention-2/weights/wisppn_P22-best.pkl')
            pck_50_overall_max = pck_50_overall


        if (epoch_index+1) % 50 == 0:
            print('the train loss for the first %.1f epoch is' % (epoch_index))
            print(train_mean_loss_iter)








