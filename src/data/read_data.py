import pandas as pd

class Data:

    def __init__(self) -> None:
        self.adv_df = None

    def read_data(self) -> tuple:
        path = '../../data/dataset/adverts.csv'
        self.adv_df = pd.read_csv(path)

        return self.adv_df.shape