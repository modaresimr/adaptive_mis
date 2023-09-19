import csv
import os

class CSVLogger:
    def __init__(self, filepath, fieldnames, separator=',', append=False):
        self.filepath = filepath
        self.separator = separator
        self.append = append
        self.fieldnames = fieldnames

        if not append or not os.path.isfile(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=separator)
                writer.writeheader()

    def log(self, metrics):
        write_mode = 'a' if self.append else 'w'
        with open(self.filepath, write_mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
            writer.writerow(metrics)