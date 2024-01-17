import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import pickle

class Regressor():
    
    def __init__(self, x, learning_rate, batch_size, epochs):
        X, _ = self._preprocessor() # X will have the returned preprocessed values.
        # Whatever other args or values that the self should contain...
        self.input_size = X.shape[1]
        self.output_size = 1
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


    def _preprocessor(self, data):

        # Filling in NaN values
        mean_total_bedrooms = data["total_bedrooms"].mean() 
        data["total_bedrooms"].fillna(mean_total_bedrooms)

        # Encode values that are not numerical -> ocean_proximity.
        features = ["latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity", "median_house_value"]
        preprop = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
        transformed_data = preprop.fit_transform(data)
        column_names = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN', 'index', 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
        transformed_df = pd.DataFrame(transformed_data, columns=column_names)
        transformed_df.drop(columns=['index'], inplace=True)

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


    def fit(self, data):
        X, Y = self._preprocessor(data=data)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, Y, train_size=0.8)

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 18),
            nn.ReLU(),
            nn.Linear(18, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        loss_function = nn.MSELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            self.model.train()
            total_training_loss = 0

            for batch_inputs, batch_target in train_loader:
                optimiser.zero_grad()
                outputs = self.model(batch_inputs)
                loss = loss_function(outputs, batch_target)
                loss.backward()
                optimiser.step()

                total_training_loss += loss.item()

        average_training_loss = loss.item() / len(train_loader)
        print(f"Average training loss for Epoch {epoch + 1} is {average_training_loss}")

        return self.model

    def predict(self):
        #Uses the model to predict, pass in testing data.
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_val)

        return predictions.numpy()

    def score(self, y):
        #Evaluates the model with the predictions given.

        true_values = self.y_val.numpy()
        predicted_values = self.predict()

        mse = mean_squared_error(true_values, predicted_values)

        return mse

        

def save_regressor(trained_model):
    #Save the model
    with open('nn_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)

    print("\n Saved model in nn_model.pickle\n")

def load_regressor():
    
    with open('nn_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
        
    print("\n Loaded model from nn_model.pickle\n")

    return trained_model

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
    transformed_df.drop(columns=['index'], inplace=True)


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

    print(x.shape)
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