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
            self.epoch_progress_bar = tqdm(total=self.params['epochs'], position=0, desc='Epoch Progress', disable=self.verbose < 2)
        self.batch_progress_bar = tqdm(total=self.params['steps'], position=1, desc='Batch Progress', disable=self.verbose < 1)

    def on_validation_begin(self, logs=None):
        self.batch_progress_bar.set_postfix_str(f"Validating... {logs or ''}")

    def on_epoch_end(self, epoch, logs=None):
        self.batch_progress_bar.set_postfix_str(logs)
        if self.verbose < 2:
            self.batch_progress_bar.close()
        self.epoch_progress_bar.set_postfix_str(logs)
        # print()
        self.epoch_progress_bar.update(1)

    def on_batch_end(self, batch, logs=None):
        self.batch_progress_bar.update(1)
        self.batch_progress_bar.set_description(logs)
