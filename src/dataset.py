from torch.utils.data import Dataset
from data_util import read_sequential_target, read_sequential_target_lang, read_sequential_target_language

class CLANT(Dataset):
    def __init__(self, dataset_dirs, test=False, remhyph=False):
        # get the dataset folders
        if test:
            lang_dir = dataset_dirs.L_dir_test
            joints_dir = dataset_dirs.B_dir_test
            vis_dir = dataset_dirs.V_dir_test
            trans_dir = dataset_dirs.T_dir_test
            trans_target_dir = dataset_dirs.T_tar_test
        else:
            lang_dir = dataset_dirs.L_dir
            joints_dir = dataset_dirs.B_dir
            vis_dir = dataset_dirs.V_dir
            trans_dir = dataset_dirs.T_dir
            trans_target_dir = dataset_dirs.T_tar

        # get the language for descriptions (L)
        self.L_fw, self.L_len, self.L_filenames = read_sequential_target_language(lang_dir, True, remhyph)

        # get the joint angles for actions (B)
        self.B_fw, self.B_bw, self.B_bin, self.B_len, self.B_filenames = read_sequential_target(joints_dir, True)
        # before normalisation save max and min joint angles to variables (will be used when converting norm to original values)
        self.maximum_joint = self.B_fw.max()
        self.minimum_joint = self.B_fw.min()

        # get the visual features for action images (V)
        self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target(vis_dir)

        # get the language for translation (T)
        self.T_fw, self.T_len, self.T_filenames = read_sequential_target_language(trans_dir, True)

        # get the language for translation testing (T_t)
        self.T_t_fw, self.T_t_len, self.T_t_filenames = read_sequential_target_language(trans_target_dir, True)

    def __len__(self):
        return len(self.L_len)

    def __getitem__(self, index):
        items = {}
        items["L_fw"] = self.L_fw[index]
        items["L_len"] = self.L_len[index] / 4     # 4 alternatives per description
        items["L_filenames"] = self.L_filenames[index]
        items["B_fw"] = self.B_fw[:, index, :]
        items["B_bw"] = self.B_bw[:, index, :]
        items["B_len"] = self.B_len[index]
        items["B_bin"] = self.B_bin[:, index, :]
        items["B_filenames"] = self.B_filenames[index]
        items["V_fw"] = self.V_fw[:, index, :]
        items["V_bw"] = self.V_bw[:, index, :]
        items["V_len"] = self.V_len[index]
        items["max_joint"] = self.maximum_joint
        items["min_joint"] = self.minimum_joint
        items["T_fw"] = self.T_fw[index]
        items["T_len"] = self.T_len[index]
        items["T_filenames"] = self.T_filenames[index]
        items["T_t_fw"] = self.T_t_fw[index]
        items["T_t_len"] = self.T_t_len[index]
        items["T_t_filenames"] = self.T_t_filenames[index]
        return items