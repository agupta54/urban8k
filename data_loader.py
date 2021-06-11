import pandas as pd

class Urban8kDataset:
    def __init__(self, metadata_file, fold_for_test):
        self.metadata_file = metadata_file
        self.fold_for_test = fold_for_test

    def train_dataset(self):
        df = pd.read_csv(self.metadata_file)
        df = df[df['classID'] != self.fold_for_test]
        return df

    def __len__(self):
        return len(self.train_dataset())

    def __getitem__(self, item):
        
