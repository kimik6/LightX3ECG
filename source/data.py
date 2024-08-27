import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        df_path, 
        augment = False, 
    ):
        
        self.chapmann_df  = pd.read_csv(df_path) 
        # self.chapmann_df = self.df[self.df['Ecg_dir'].str.contains('chapmanshaoxing', na=False)]

    def __len__(self, 
    ):
        return len(self.chapmann_df)


    def __getitem__(self, 
        index, 
    ):
        row = self.chapmann_df.iloc[index]

#         ecg = loadmat("{}/{}".format('/kaggle/input', row["Ecg_dir"]))
        ecg, header_data = load_challenge_data("{}/{}".format('/kaggle/input', row["Ecg_dir"]))
#         ecg = pad_sequences(ecg, "float64", 
#             "post", "post", 
#         )

        ecg = torch.tensor(ecg[:,:5000]).float()
        
        label = row["bLabs0"]
        
        return ecg, label