from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split


class Split:
    def __init__(self, dataset, cfg_dataloader, train_size=0.5, val_size=0.2, test_size=0.3, seed=42, **kwargs):
        self.dataset = dataset
        self.cfg_dataloader = cfg_dataloader
        self.test_size = test_size
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed

    def next(self):
        train_val_dataset, test_dataset = train_test_split(self.dataset, test_size=self.test_size, random_state=self.seed)
        test_loader = DataLoader(test_dataset, **self.cfg_dataloader['test'])
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=self.val_size, random_state=self.seed)
        train_loader = DataLoader(train_dataset, **self.cfg_dataloader['train'])
        val_loader = DataLoader(val_dataset, **self.cfg_dataloader['validation'])

        print(f"~~~~~~~~~~~~~~~ Fold {0} ~~~~~~~~~~~~~~~~~")
        print(f"Length of trainig_dataset:\t{len(train_dataset)}")
        print(f"Length of validation_dataset:\t{len(val_dataset)}")
        print(f"Length of test_dataset:\t\t{len(test_dataset)}")
        yield train_loader, val_loader, test_loader, 0
