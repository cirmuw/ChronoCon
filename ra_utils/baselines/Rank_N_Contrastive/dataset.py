import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data

### Hard coded inputs  (for now)
# PATH_TO_DATA_TABLE = "/home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_RnC_paper_maybe_modified.csv"
# PATH_TO_DATA_TABLE = "/home/cwatzenboeck/data/public/agedb/tabular/agedb_splits_stratified_new.csv"


PATH_TO_DATA_TABLE = "/home/cwatzenboeck/data/public/agedb/tabular/03_agedb_splits_stratified_new.csv"

class AgeDB(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train', path_to_data_table=PATH_TO_DATA_TABLE):
        df = pd.read_csv(path_to_data_table, encoding='utf-8')
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.split = split
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sample = {}
        sample['y_true'] = np.asarray([row['age']]).astype(np.float32)
        sample['name'] = row['name'] if 'name' in row else None
        #sample['sex'] = row['sex'] if 'sex' in row else None
        #sample['image_id'] = row['image_id'] if 'image_id' in row else None
        #sample['path'] = row['path']
        img_path = os.path.join(self.data_folder, row['path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        sample['image'] = img
        return sample
