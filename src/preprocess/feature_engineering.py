import pandas as pd
from datetime import date
import constants as c

class Feature_engineering:


    def __init__(self, obj_data) -> None:
        self.obj_data = obj_data

    def new_ftr_age(self) -> pd.DataFrame:
        # Adding new feature age by subtracting current year from year of registration.
        self.obj_data.adv_df[c.COL_AGE] = date.today().year - self.obj_data.adv_df[c.COL_REG_YEAR]
        # Deleting feature year of registration
        return self.obj_data.adv_df.drop(columns=[c.COL_REG_YEAR])

    def ftr_engineering(self):
        self.obj_data.adv_df = self.new_ftr_age()

        