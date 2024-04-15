import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_sequential_target
from config import TrainConfig

def evaluate(signal='describe', app_length=True, checkpoint_name='cross_t5.tar', robustness_mode=0):
    # get the training configuration
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")

    # read dataset
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir, True)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True)
    max_joint = np.concatenate((B_fw, B_fw_u), 1).max()
    min_joint = np.concatenate((B_fw, B_fw_u), 1).min()
    # prepare training data
    B_fw = B_fw.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    # prepare test data
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))

    # get predictions from the interference run
    predict_train, _, predtrain_bin, predtrain_len, _ = read_sequential_target('../train/inference/' + checkpoint_name.lstrip('/').rstrip('.tar') + '_robustness/' + str(robustness_mode) + "/" + signal + '/prediction/behavior_train/', True)
    predict_test, _, predtest_bin, predtest_len, _ = read_sequential_target('../train/inference/' + checkpoint_name.lstrip('/').rstrip('.tar') + '_robustness/' + str(robustness_mode) + "/" + signal + '/prediction/behavior_test/', True)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predtrain_bin = predtrain_bin.transpose((1,0,2))
    predtest_bin = predtest_bin.transpose((1,0,2))
    predict_train = predict_train * B_bin[:, 1:, :]
    predict_test = predict_test * B_bin_u[:, 1:, :]
    # set target depending on signal
    if signal == 'describe': # take last joint configuration as target
        gt = np.repeat(np.expand_dims(B_bw_u[:, 0, :], 1), B_bw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    elif signal == 'execute' or signal =='repeat action':
        gt = B_fw_u[:,1:,:]
    else:
        gt = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1), B_fw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    # if appriori length is false, fill shorter prediction with zeros
    if app_length == False:
        if gt.shape[1] > predict_test.shape[1]:
            predict_test = np.concatenate((predict_test, np.zeros((predict_test.shape[0], gt.shape[1] - predict_test.shape[1], predict_test.shape[2]))), 1)

    # calculate error
    mse = np.mean(np.square(predict_test - gt))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_joint - min_joint)
    nrmse_test = nrmse * 100

    if signal == 'describe':
        gt_t = np.repeat(np.expand_dims(B_bw[:, 0, :], 1), B_bw.shape[1] - 1, axis=1) * B_bin[:, 1:, :]
    elif signal == 'execute' or signal =='repeat action':
        gt_t = B_fw[:,1:,:]
    else:
        gt_t = np.repeat(np.expand_dims(B_fw[:, 0, :], 1), B_fw.shape[1] - 1, axis=1) * B_bin[:, 1:, :]
    # if appriori length is false, fill shorter prediction with zeros
    if app_length == False:
        if gt_t.shape[1] > predict_train.shape[1]:
            predict_train = np.concatenate((predict_train, np.zeros((predict_train.shape[0], gt_t.shape[1] - predict_train.shape[1], predict_train.shape[2]))), 1)

    # calculate error
    mse = np.mean(np.square(predict_train - gt_t))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_joint - min_joint)
    nrmse_train = nrmse * 100

    return nrmse_train, nrmse_test

if __name__ == '__main__':
    nrmse_train, nrmse_test = evaluate('repeat action', True)
    # print error
    print('Normalised Root-Mean squared error (NRMSE) for predicted train joint values, in percent: ', nrmse_train)
    print('Normalised Root-Mean squared error (NRMSE) for predicted test joint values, in percent: ', nrmse_test)