import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import torch

class Regressor():
    
    def __init__(self, x):
        X, _ = self._preprocessor() # X will have the returned preprocessed values.
        # Whatever other args or values that the self should contain...

    def _preprocessor(self, data, training=False):

        # Filling in NaN values
        mean_total_bedrooms = data["total_bedrooms"].mean() 
        data["total_bedrooms"].fillna(mean_total_bedrooms)

        # Encode values that are not numerical -> ocean_proximity.
        features = ["latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity", "median_house_value"]
        preprop = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
        transformed_data = preprop.fit_transform(data)
        column_names = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'index', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
        transformed_df = pd.DataFrame(transformed_data, columns=column_names)

        train_split_index = int(0.8 * len(transformed_df))
        test_split_index = len(transformed_df) - train_split_index

        output_label = 'median_house_value'

        self.x = transformed_df.loc[: , transformed_df.columns != output_label]
        self.y = transformed_df.loc[: , transformed_df.columns == output_label]

        x_tensor = torch.tensor(self.x.values, dtype=torch.float32)
        y_tensor = torch.tensor(self.y.values, dtype=torch.float32)

        x_tensor = (x_tensor - x_tensor.min(dim=0).values / (x_tensor.max(dim=0).values - x_tensor.min(dim=0).values))
        y_tensor = (y_tensor - y_tensor.min(dim=0).values / (y_tensor.max(dim=0).values - y_tensor.min(dim=0).values))

        return x_tensor, y_tensor

        # Normalize our data

        # self.x_train = transformed_df.loc[:train_split_index, transformed_df.columns != output_label]
        # self.y_train = transformed_df.loc[:train_split_index, transformed_df.columns == output_label]

        # self.x_test = transformed_df.loc[train_split_index+1: , transformed_df.columns != output_label]
        # self.y_test = transformed_df.loc[train_split_index+1: , transformed_df.columns == output_label]
            
        # test_tensor_x = torch.tensor(self.x_test, dtype=torch.float32)
        # train_tensor_x = torch.tensor(self.x_train, dtype=torch.float32)
        # test_tensor_y = torch.tensor(self.y_test, dtype=torch.float32)
        # train_tensor_y = torch.tensor(self.y_train, dtype=torch.float32)


    def fit(self, x ,y):
        #Fits the model to the data...
        pass

    def predict(self, x):
        #Uses the model to predict
        pass

    def score(self, y):
        #Evaluates the model
        pass

def save_regressor(trained_model):
    #Save the model
    pass

def load_regressor():
    #Load from pickle file
    pass

def HyperParameterSearch(X, y, model):
    pass

def example_main():

    data = pd.read_csv('housing.csv')
    data = shuffle(data).reset_index()
    # print(data)
    mean_total_bedrooms = data["total_bedrooms"].mean()
    data["total_bedrooms"].fillna(mean_total_bedrooms)

    # Split into train and test... --> we'll choose 80/20 split
    features = ["latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity", "median_house_value"]

    preprop = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')

    transformed_data = preprop.fit_transform(data)

    column_names = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'index', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
    transformed_df = pd.DataFrame(transformed_data, columns=column_names)


    # rename_dict = {'onehotencoder__ocean_proximity_<1H OCEAN': '<1H OCEAN', 'onehotencoder__ocean_proximity_INLAND': 'INLAND', 'onehotencoder__ocean_proximity_ISLAND': 'ISLAND', 'onehotencoder__ocean_proximity_NEAR BAY': 'NEAR BAY', 'onehotencoder__ocean_proximity_NEAR OCEAN': 'NEAR OCEAN'}
    # transformed_df.rename(columns=rename_dict, inplace=True)

    # print(transformed_df)

    # print(transformed_df)
    # print(transformed_df.columns)
    
    train_split_index = int(0.8 * len(transformed_df))
    test_split_index = len(transformed_df) - train_split_index

    output_label = 'median_house_value'

    x = transformed_df.loc[:, transformed_df.columns != output_label]
    y = transformed_df.loc[:, transformed_df.columns == output_label]

    # print(data)
    # x_train = transformed_df.loc[:train_split_index, transformed_df.columns != output_label]
    # y_train = transformed_df.loc[:train_split_index, transformed_df.columns == output_label]

    # x_test = transformed_df.loc[train_split_index+1: , transformed_df.columns != output_label]
    # y_test = transformed_df.loc[train_split_index+1: , transformed_df.columns == output_label]

    # print(x_test.values)

    # print(type(tensor_x))
    # print(len(x_train) + len(x_test) == len(data))
    # print(train_split_index, test_split_index)
    # regressor = Regressor()


if __name__ == "__main__":
    example_main()