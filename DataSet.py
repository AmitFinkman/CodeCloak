import pandas as pd
import random

from torch.utils.data import Dataset


class PromptsDataset(Dataset):
    """prompts and responses data set"""

    def __init__(self, csv_file, sub=None):
        """
        :param csv_file (string): Path to the csv file with annotations
        """
        prompts_data = pd.read_csv(csv_file)
        results_df = prompts_data
        if sub is None:
            self.prompts_responses_csv = results_df
        else:
            self.prompts_responses_csv = results_df.head(sub)

    def __len__(self):
        return len(self.prompts_responses_csv)

    def __getitem__(self, idx):
        sample = self.prompts_responses_csv.iloc[idx].to_dict()
        for k, v in sample.items():
            if pd.isna(v):
                sample[k] = ''

        return sample

    def get_random_sample(self):
        idx = random.randrange(0, self.__len__())
        return self.__getitem__(idx), idx





