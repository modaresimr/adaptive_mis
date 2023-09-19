from tqdm import tqdm

class TqdmCallback:
    def __init__(self, verbose=1):
        self.verbose = verbose

    def set_params(self, params):
        self.params = params
        self.epoch_progress_bar = None
        self.batch_progress_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch_progress_bar is None:
            self.epoch_progress_bar = tqdm(total=self.params['epochs'], position=0, desc='Epoch Progress', disable=self.verbose == 0)
        self.batch_progress_bar = tqdm(total=self.params['steps'], position=1, desc='Batch Progress', disable=self.verbose == 0)

    def on_epoch_end(self, epoch, logs=None):
        self.batch_progress_bar.close()
        self.epoch_progress_bar.update(1)

    def on_batch_end(self, batch, logs=None):
        self.batch_progress_bar.update(1)
