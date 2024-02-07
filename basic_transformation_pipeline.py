import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

#from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('<PATH>', 'preprocessor.pkl') # Path to save preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function transforms the data
        '''
    # Define categorical and numeric features
        categorical_features = [] # Enter categorical features
        numeric_features = [] # Enter numeric features
        binary_features = [] # Enter binary features

        # Numeric transformer pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
            ('scaler', StandardScaler())
        ])

        # Categorical transformer pipeline
        categorical_transformer = Pipeline(steps=[
            ('OneHotEncoder', OneHotEncoder())
        ])

        # Combine numeric and categorical transfomer pipelines
        preprocessor = ColumnTransformer(
            remainder = 'passthrough',
            transformers=[
                ('categorical', categorical_transformer, categorical_features),
                ('numeric', numeric_transformer, numeric_features)
        ])

        # Smote transformer to handle unbalanced data
        smote_processor = SMOTE(random_state=865)

        return preprocessor, smote_processor

        
    def initiate_data_transformation(self, train_path, test_path):

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessing_obj, smote_obj = self.get_data_transformer_object()

        target_column_name = '' # Enter target column
        id_column = '' # Enter ID column

        input_feature_train_df = train_df.drop(columns=[target_column_name, id_column], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=[target_column_name, id_column], axis=1)
        target_feature_test_df = test_df[target_column_name]


        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        input_feature_train_smote_arr, input_target_train_smote_arr = smote_obj.fit_resample(input_feature_train_arr, target_feature_train_df)

        train_arr = np.c_[input_feature_train_smote_arr, np.array(input_target_train_smote_arr)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        #save_object(
        #    file_path=self.data_transformation_config.preprocessor_obj_file_path,
        #    obj=preprocessing_obj
        #)

        return(
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
        )
