import pandas as pd
import numpy as np
import constants as c

class Data_preprocess:


    def __init__(self, obj_data) -> None:
        self.obj_data = obj_data

    
    def na_count(self) -> list[str]:
        # Locating missing data, replacing them.
        # Get the count of NaN values in each column & iterate through the counts and find columns with NaN values
        na_counts = self.obj_data.adv_df.isna().sum()
        return [col for col, count in na_counts.items() if count != 0]
    

    def rm_noise(self) -> None:
        # Removing rows for associated reg_code values that have been identified as noise.
        self.obj_data.adv_df.drop(self.obj_data.adv_df[(self.obj_data.adv_df[c.COL_REG_CODE].isna() == True) & 
                                                       (self.obj_data.adv_df[c.COL_VHL_COND] != 'NEW')].index, inplace=True)
        

    def rm_na_reg_code(self):
        # Replacing the null values in 'reg_code' column where vehicle condition is 'NEW'.
        self.obj_data.adv_df.loc[(self.obj_data.adv_df[c.COL_REG_CODE].isna() == True) & 
                                 (self.obj_data.adv_df[c.COL_VHL_COND] == 'NEW'), c.COL_REG_CODE] = '71'
        

    def rm_na_reg_year(self):
        # Replacing null values in 'year_of_registration' base
        self.obj_data.adv_df[c.COL_REG_YEAR] = \
            np.where((self.obj_data.adv_df[c.COL_REG_YEAR].isna()) & 
                        (self.obj_data.adv_df[c.COL_REG_CODE] < 50), 2000 + self.obj_data.adv_df[c.COL_REG_CODE], 
                        2000 + self.obj_data.adv_df[c.COL_REG_CODE] - 50)
        

    def rm_na_mileage(self):
        yr_mlg_dict = self.obj_data.adv_df.groupby('year_of_registration').mean().round(3)['mileage'].to_dict()

        for index, row in self.obj_data.adv_df[(self.obj_data.adv_df[c.COL_MILEAGE].isna()) & 
                                                (self.obj_data.adv_df[c.COL_REG_YEAR].isna() == False)].iterrows():
            self.obj_data.adv_df.at[index, c.COL_MILEAGE] = yr_mlg_dict[row[c.COL_REG_YEAR]]


    def rm_na_colr(self):
        mode_clr = self.obj_data.adv_df[c.COL_STD_COLR].mode()[0]
        self.obj_data.adv_df[c.COL_STD_COLR].fillna(mode_clr, inplace = True)


    def rm_na_bd_type(self):
        mode_body_type = self.obj_data.adv_df[c.COL_BD_TYPE].mode()[0]
        self.obj_data.adv_df[c.COL_BD_TYPE].fillna(mode_body_type, inplace = True)


    def rm_na_fl_type(self):
        mode_fuel_type = self.obj_data.adv_df[c.COL_FL_TYPE].mode()[0]
        self.obj_data.adv_df[c.COL_FL_TYPE].fillna(mode_fuel_type, inplace = True)


    def rm_missing_val(self) -> str:
        for col in self.na_count():
            if col == c.COL_REG_CODE:
                self.rm_na_reg_code()

            if col == c.COL_REG_YEAR:
                self.rm_na_reg_year()
            
            if col == c.COL_MILEAGE:
                self.rm_na_mileage()
                
            if col == c.COL_STD_COLR:
                self.rm_na_colr()

            if col == c.COL_BD_TYPE:
                self.rm_na_bd_type()

            if col == c.COL_FL_TYPE:
                self.rm_na_fl_type()
        return 'success'
    

    def rm_outlr_mlg(self):
        Q3 = np.quantile(self.obj_data.adv_df[c.COL_MILEAGE], 0.75)
        Q1 = np.quantile(self.obj_data.adv_df[c.COL_MILEAGE], 0.25)
        IQR = Q3 - Q1

        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        outlr_free_lst = [x for x in self.obj_data.adv_df[c.COL_MILEAGE] if ( (x > lower_range) & (x < upper_range))]
        self.obj_data.adv_df = self.obj_data.adv_df.loc[self.obj_data.adv_df[c.COL_MILEAGE].isin(outlr_free_lst)]


    def to_num(self) -> None:
        # Converting the reg_code values to numeric type.
        self.obj_data.adv_df[c.COL_REG_CODE] = pd.to_numeric(self.obj_data.adv_df[c.COL_REG_CODE])


    def preprocess(self):
        # Detect and deal with noise, missing values, and outliers.
        self.rm_noise()
        self.rm_missing_val()
        self.rm_outlr_mlg()
        self.to_num()
