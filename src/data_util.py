import os
import time
import numpy as np
import torch


def get_file_list(path):
  all_file_list = []
  dir_list = []
  for (root, dirs, files) in os.walk(path):
    dir_list.append(root)
    file_list = []
    for file in files:
      file_list.append(os.path.join(root, file))
    file_list.sort()
    all_file_list.append(file_list)
  return all_file_list, dir_list

# read torch pickle files
def read_sequential_target_pickle(root_path, with_filename=False):
  all_file_list, _ = get_file_list(root_path)
  all_file_list.sort()
  data_num = 0
  max_len = 0

  load_file = torch.load

  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      data_num += 1
      data = load_file(file, map_location=torch.device('cpu'))
      if data.shape[0] > max_len:
        max_len = data.shape[0]

  fw = np.zeros((data_num, max_len, data.shape[1]))
  bw = np.zeros((data_num, max_len, data.shape[1]))
  binary = np.zeros((data_num, max_len, data.shape[1]))
  length = []

  count = 0
  filenames = []
  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      # print file
      filenames.append(file)
      data = load_file(file)
      fw[count, :data.shape[0], :] = data.numpy()
      bw[count, :data.shape[0], :] = data.numpy()[::-1]
      binary[count, :data.shape[0], :] = 1.0
      length.append(data.shape[0])
      count += 1

  fw = fw.transpose((1, 0, 2))
  bw = bw.transpose((1, 0, 2))
  binary = binary.transpose((1, 0, 2))
  length = np.array(length)
  if with_filename:
    return fw, bw, binary, length, filenames
  else:
    return fw, bw, binary, length

# read the dataset txt files
def read_sequential_target(root_path, with_filename=False, is_npy=False, max_len=None):
  all_file_list, _ = get_file_list(root_path)
  all_file_list.sort()
  data_num = 0
  if max_len == None:
    max_len = 0
  if is_npy:
    load_file = np.load
  else:
    load_file = np.loadtxt

  # Goes through all files in the target folder to read out their sample count, max length and shape
  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      data_num += 1
      data = load_file(file)
      if len(data.shape) == 1:
        data = np.expand_dims(data, 0)
      if data.shape[0] > max_len:
        max_len = data.shape[0]
  fw = np.zeros((data_num, max_len, data.shape[1])) # Creates initial zero array for fw
  bw = np.zeros((data_num, max_len, data.shape[1])) # Creates initial zero array for bw
  binary = np.zeros((data_num, max_len, data.shape[1])) # Creates initial zero array for bin
  length = []
  
  count = 0
  filenames = []
  # Goes through all files in the target folder to get the values
  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      #print file
      filenames.append(file)
      data = load_file(file)
      fw[count, :data.shape[0], :] = data # Writes values from current file to corresponding fw array index
      bw[count, :data.shape[0], :] = data[::-1] # Writes values from current file to corresponding bw array index in opposite order
      binary[count, :data.shape[0], :] = 1.0 # Fills binary with ones everywhere where a value is in the file
      length.append(data.shape[0])
      count += 1

  fw = fw.transpose((1,0,2))
  bw = bw.transpose((1,0,2))
  binary = binary.transpose((1,0,2))
  length = np.array(length)
  if with_filename:
    return fw, bw, binary, length, filenames
  else:
    return fw, bw, binary, length


# read the dataset txt files
def read_sequential_target_lang(root_path, with_filename=False):
  all_file_list, _ = get_file_list(root_path)
  all_file_list.sort()
  fw = []
  filenames = []
  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      # print file
      filenames.append(file)
      with open(file) as f:
        data = f.readline().strip('\n')
      fw.append(data)

  if with_filename:
    return fw, filenames
  else:
    return fw

# read the dataset txt files, document length and optionally remove hyphens
# hyphens are included in the NICO dataset descriptions, but T5 works better without them
def read_sequential_target_language(root_path, with_filename=False, remhyph=False):
  all_file_list, _ = get_file_list(root_path)
  all_file_list.sort()
  fw = []
  filenames = []
  length = []
  for file_list in all_file_list:
    file_list.sort()
    for i, file in enumerate(file_list):
      # print file
      filenames.append(file)
      with open(file) as f:
        data = f.read().splitlines()
      length.append(len(data))
      if remhyph == True:
        data = [d.replace('-', ' ') for d in data]
      fw.append(data)

  if with_filename:
    return fw, length, filenames
  else:
    return fw, length

# save the binding layer features as txt files
def save_latent(c, name, dirname="latent"):
  dir_hierarchy = name.split("/")
  dir_hierarchy[1] = 'prediction'
  dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
  save_name = os.path.join(*dir_hierarchy)
  dirname = '../train/' + dirname
  save_name = os.path.join(dirname, save_name)
  if not os.path.exists(os.path.dirname(save_name)):
    os.makedirs(os.path.dirname(save_name))
  np.savetxt(save_name, c[0], fmt="%.6f")

# save the binding layer features as txt files
def save_text(c, name, dirname="latent"):
  dir_hierarchy = name.split("/")
  dir_hierarchy[1] = 'prediction'
  dir_hierarchy = filter(lambda z: z != "..", dir_hierarchy)
  save_name = os.path.join(*dir_hierarchy)
  dirname = '../train/' + dirname
  save_name = os.path.join(dirname, save_name)
  if not os.path.exists(os.path.dirname(save_name)):
    os.makedirs(os.path.dirname(save_name))
  with open(save_name, 'w') as f:
    f.write(c[0])

class Logger():
  def __init__(self):
    self.total_time = 0
    self.start_time = time.time()
    self.error_arr = np.zeros((0))
  def __call__(self, epoch, loss):
    current_time = time.time()
    self.total_time = current_time - self.start_time
    print("epoch:{} time:{} LOSS: {}".format(epoch, self.total_time, loss))
    self.error_arr = np.r_[self.error_arr, loss]

def normalise(data, max, min):
  data = 2 * ((data - min) / (max - min)) - 1
  return data

def reverse_normalise(data, max, min):
  data = ((data + 1) / 2) * (max - min) + min
  return data

# Don't use this function. It's dataset specific. Multiply by bin matrices instead.
def pad_with_zeros(data, test=False):
      if test:
          for i in range(6):
              data[50:, 18 + 36 * i: 36 + 36 * i, :] = 0
      else:
          for i in range(6):
              data[50:, 54 + 108 * i: 108 + 108 * i, :] = 0
      return data

def add_active_feature(joints, active):
  active_feat_dim = np.expand_dims(active[:,:,0], -1)
  joints = np.concatenate((joints,active_feat_dim), -1)
  return joints