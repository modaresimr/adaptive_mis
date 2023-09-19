from copy import deepcopy
class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', verbose=False, patience=10, restore_best_weights=False):
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, metrics, model):
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            raise ValueError(f"Monitor '{self.monitor}' not found in metrics")

        if self.best_score is None:
            self.best_score = current_score

            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())

        elif ((self.mode == 'min' and current_score < self.best_score) or 
              (self.mode == 'max' and current_score > self.best_score)):
            self.best_score = current_score
            self.counter = 0

            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

                if self.restore_best_weights:
                    print("Restoring model weights to the best weights.")
                    model.load_state_dict(self.best_weights)
