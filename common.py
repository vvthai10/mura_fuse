import os

import torch

# output =  "/content/drive/MyDrive/Thesis/Sources/storages/mura-fuse"
output =  "./output"

class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = "lqn_mura_v4"
    data_dir = "./data/"
    output_dir = output
    exp_dir = os.path.join(output_dir, exp_name)
    log_dir = os.path.join(exp_dir, "log/")
    model_dir = os.path.join(exp_dir, "model/")
    study_type = [
        # Original
        "ELBOW",
        "FINGER",
        "FOREARM",
        "HAND",
        "HUMERUS",
        "SHOULDER",
        "WRIST",
        # New
        "FEMUR",
        "LEG",
        "KNEE",
    ]

    acc_path = os.path.join(exp_dir, "acc.csv")

    def make_dir(self):
        self.exp_dir = os.path.join(output, self.exp_name)
        if not os.path.exists(self.exp_dir):
            os.makedirs(os.path.join(self.exp_dir, "model"))
            os.makedirs(os.path.join(self.exp_dir, "log"))
        self.log_dir = os.path.join(self.exp_dir, "log/")
        self.model_dir = os.path.join(self.exp_dir, "model/")


config = Config()
