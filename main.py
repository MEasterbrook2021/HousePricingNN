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
    
    def __init__(self, data, learning_rate, batch_size, epochs): # Learning rate, batch size and epochs will be hyperparameters to tune later.
        X, _ = self._preprocessor(data) # X will have the returned preprocessed values.

        self.input_size = X.shape[1]    # This will be our input node to the Neural Network
        self.output_size = 1    # Linear regression, just want a number -> output size of 1.

        # Hyperparameters that will be tuned using the hyperparameter tuning function
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.epochs = epochs


    def _preprocessor(self, data):
        mean_total_bedrooms = data["total_bedrooms"].mean() # Filling in NaN values, in this case only mean_total_bedrooms...
        data["total_bedrooms"].fillna(mean_total_bedrooms, inplace=True) # Should make a case for general NaN values.
        

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

        # print("Checking preprocessor for NaN values: ", self.x.isna().sum())

        x_tensor = torch.tensor(self.x.values, dtype=torch.float32)
        y_tensor = torch.tensor(self.y.values, dtype=torch.float32)

        x_tensor = (x_tensor - x_tensor.min(dim=0).values / (x_tensor.max(dim=0).values - x_tensor.min(dim=0).values))
        y_tensor = (y_tensor - y_tensor.min(dim=0).values / (y_tensor.max(dim=0).values - y_tensor.min(dim=0).values))

        return x_tensor, y_tensor


    def fit(self, data, validation=False): # Only needs training data...
        X, Y = self._preprocessor(data=data)

        if(validation==False):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=0.8)
        else:
            self.X_train, X_temp, self.y_train, y_temp = train_test_split(X, Y, train_size=0.7)
            self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5)


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

        # average_training_loss = loss.item() / len(train_loader)
        # print(f"Average training loss for Epoch {epoch + 1} is {average_training_loss}")

        return self.model

    def predict(self, validation=False):
        self.model.eval()
        with torch.no_grad():
            if validation:
                predictions = self.model(self.X_val)
            else:
                predictions = self.model(self.X_test)

        return predictions.numpy()

    def score(self, validation=False): # Needs to take in data as an argument.
        if validation:
            true_values = self.y_val.numpy()
            predicted_values = self.predict(validation=True)
        else:
            true_values = self.y_test.numpy() # This is y_true
            predicted_values = self.predict() # Predicted is obtained from X_test

        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)

        return rmse



def save_regressor(trained_model):
    with open('nn_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)

    print("\n Saved model in nn_model.pickle\n")

def load_regressor():
    with open('nn_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
        
    print("\n Loaded model from nn_model.pickle\n")

    return trained_model

def HyperParameterSearch(model, params, data):
    # Split into 70/15/15 training/validation/testing
    # Then score with the final 15%?
    # Want to store best_model and cur_model

    best_score = 9999999
    cur_score = 0
    best_params = {"learning_rate" : 0,
                   "nb_epochs" : 0,
                   "batch_size" : 0}

    for rate in params["learning_rates"]:
        print("\n Trying {} as learning rate".format(rate))
        model.learning_rate = rate
        for epochs in params["nb_epochs"]:
            print("\n Trying {} epochs".format(epochs))
            model.nb_epochs = epochs
            for batch_size in params["batch_size"]:
                print("\n Trying {} batch size".format(batch_size))
                model.batch_size = batch_size
                model.fit(data=data, validation=True)
                cur_score = model.score(validation=True)
                if(cur_score < best_score):
                    best_params["learning_rate"], best_params["batch_size"], best_params["nb_epochs"] = rate, batch_size, epochs
                    best_score = cur_score
    
    print("\n Best RMSE from hyperparameter tuning is: {}\n".format(best_score))

    return best_params

def example_main():
    HyperParamTuning = True

    data = pd.read_csv('housing.csv')
    data = shuffle(data).reset_index()

    regressor = Regressor(data=data, learning_rate=0.001, batch_size=16, epochs=10)

    params = {"learning_rates" : [0.001, 0.002, 0.005, 0.01, 0.02],
              "nb_epochs" : [50, 100, 200],
              "batch_size" : [8, 16, 32]}

    if not HyperParamTuning: 

        regressor.fit(data=data, validation=False)
        error = regressor.score()

        print("\n Regressor error is: {}\n".format(error))

    else:
        regressor.fit(data=data, validation=True)
        best_parameters = HyperParameterSearch(data=data, model=regressor, params=params)

        print("Best parameters are", best_parameters)
    
if __name__ == "__main__":
    example_main()