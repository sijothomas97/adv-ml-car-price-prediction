from sklearn.preprocessing import OrdinalEncoder
import constants as c

class Encoder():

    def __init__(self, obj_data) -> None:
        self.obj_data = obj_data

    def encd_colr(self):
        colour_encoder = OrdinalEncoder()
        colour_encoder.fit(self.obj_data.adv_df[[c.COL_STD_COLR]])

        self.obj_data.adv_df[[c.COL_STD_COLR]] = colour_encoder.transform(self.obj_data.adv_df[[c.COL_STD_COLR]])

    def encd_make(self):
        make_encoder = OrdinalEncoder()
        make_encoder.fit(self.obj_data.adv_df[[c.COL_STD_MK]])

        self.obj_data.adv_df[[c.COL_STD_MK]] = make_encoder.transform(self.obj_data.adv_df[[c.COL_STD_MK]])

    def encd_mdl(self):
        model_encoder = OrdinalEncoder()
        model_encoder.fit(self.obj_data.adv_df[[c.COL_STD_MDL]])

        self.obj_data.adv_df[[c.COL_STD_MDL]] = model_encoder.transform(self.obj_data.adv_df[[c.COL_STD_MDL]])

    def encd_vhl_cond(self):
        self.obj_data.adv_df[c.COL_VHL_COND].replace({c.VAL_USED:0,c.VAL_NEW: 1}, inplace =True)

    def encd_bd_type(self):
        body_encoder = OrdinalEncoder()
        body_encoder.fit(self.obj_data.adv_df[[c.COL_BD_TYPE]])

        self.obj_data.adv_df[[c.COL_BD_TYPE]] = body_encoder.transform(self.obj_data.adv_df[[c.COL_BD_TYPE]])

    def encd_vhl_type(self):
        self.obj_data.adv_df[c.COL_VHL_TYPE] = self.obj_data.adv_df[c.COL_VHL_TYPE].astype(int)

    def encd_fl_type(self):
        self.obj_data.adv_df = pd.get_dummies(self.obj_data.adv_df, columns = [c.COL_FL_TYPE])

        self.obj_data.adv_df.columns = map(str.lower, self.obj_data.adv_df.columns)
        self.obj_data.adv_df.columns = self.obj_data.adv_df.columns.str.replace(' ', '_')
        self.obj_data.adv_df.columns = self.obj_data.adv_df.columns.str.replace('-', '_')

        # Deleting fuel type columns which have only least number of entries
        self.obj_data.adv_df = self.obj_data.adv_df.drop(columns=c.FL_TYPE_LST)
        


    def encoder(self):
        self.encd_colr()
        self.encd_make()
        self.encd_mdl()
        self.encd_vhl_cond()