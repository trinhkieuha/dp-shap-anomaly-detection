# Import relevant packages
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings
import numpy as np
import joblib

# Config
pd.set_option('display.max_columns', None) # Ensure all columns are displayed
warnings.filterwarnings("ignore")

def fill_data(var, data, var_info, fill_tp=None, fill_val=0):
    if fill_tp:
        fill_val = var_info[var_info["var_name"]==var][fill_tp].tolist()[0]
    return data[var].fillna(fill_val)

class ScaleData:

    def __init__(self, var_info):
        # Get list of numerical variables
        self.num_vars = var_info[var_info['var_type']=="numerical"]['var_name'].tolist()

    def scale_fit(self, train_data):
        # Create the pipeline to scale data: normalize then scale to [0,1]
        pipeline = Pipeline([('normalizer', Normalizer()),
                        ('scaler', MinMaxScaler())])
        pipeline.fit(train_data[self.num_vars])
        # Save the pipeline
        joblib.dump(pipeline, "../data/scalers")
        
        return pipeline

    def scale_transform(self, data, pipeline):
        # Isolate numerical and categorical data
        num_data = data[self.num_vars]
        cat_data = data[[col for col in data.columns if col not in self.num_vars]].reset_index(drop=True)
        # Scale the numerical data
        scaled_data = pd.DataFrame(pipeline.transform(num_data), columns=num_data.columns)
        # Concat the numerical and categorical data
        return pd.concat([scaled_data, cat_data], axis=1)