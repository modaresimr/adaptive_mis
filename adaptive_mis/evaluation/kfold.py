from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold as SKFold


class KFold:
    def __init__(self, dataset, cfg_dataloader, k=5, test_size=0.2, seed=42, **kwargs):
        self.dataset = dataset
        self.cfg_dataloader = cfg_dataloader
        self.k = k
        self.test_size = test_size
        self.seed = seed

    def count(self):
        return self.k

    def next(self):
        train_val_dataset, test_dataset = train_test_split(self.dataset, test_size=self.test_size, random_state=self.seed)
        test_loader = DataLoader(test_dataset, **self.cfg_dataloader['test'])
        # Define the number of splits for k-fold cross-validation
        kfold = SKFold(n_splits=self.k, shuffle=True)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_dataset)):
            train_dataset = Subset(train_val_dataset, train_ids)
            val_dataset = Subset(train_val_dataset, val_ids)
            train_loader = DataLoader(train_dataset, **self.cfg_dataloader['train'])
            val_loader = DataLoader(val_dataset, **self.cfg_dataloader['validation'])
            print(f"~~~~~~~~~~~~~~~ Fold {fold}/{self.k} ~~~~~~~~~~~~~~~~~")
            print(f"Length of trainig_dataset:\t{len(train_dataset)}")
            print(f"Length of validation_dataset:\t{len(val_dataset)}")
            print(f"Length of test_dataset:\t\t{len(test_dataset)}")

            yield train_loader, val_loader, test_loader, fold
