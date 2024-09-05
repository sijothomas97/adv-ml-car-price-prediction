from sklearn.model_selection import train_test_split
import constants as c


class Train:

    def __init__(self, obj_data) -> None:
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.obj_data = obj_data

    def train_test_split(self):
        X = self.obj_data.adv_df.drop(columns=c.COL_PRED)
        y = self.obj_data.adv_df[c.COL_PRED]

        return train_test_split(X, y, test_size=0.25, random_state=42)
    
    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split()
