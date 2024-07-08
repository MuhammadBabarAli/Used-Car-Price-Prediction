from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from FeatureEncoder import featureEncoder, featureEncoderUserInput
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

user_input_df = {
    'make': 'Toyota',
    'model': 'Corolla',
    'year': 2006,
    'transmission': 'Automatic',
    'fuel': 'Petrol',
    'city': 'Lahore',
    'mileage': 10000,
}

# Load the dataset
dataset = pd.read_csv('pakwheels_used_car_data_v02.csv')

# Use only relevant columns
dataset = dataset[['make', 'model', 'year', 'transmission', 'fuel', 'city', 'price', 'mileage']]
# Drop missing values
dataset = dataset.dropna()

# Encoding categorical features
categorical_features = ['make', 'model', 'transmission', 'fuel', 'city']
dataset, encoding_maps = featureEncoder(categorical_features, dataset)
user_input_df = featureEncoderUserInput(encoding_maps, user_input_df)

user_input_df = pd.DataFrame([user_input_df])


def linearRegressionPkg(data, user_input):
    # Splitting the dataset into features and target variable
    X = data.drop(['price'], axis=1)
    y = data['price']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred = linear_model.predict(X_test)

    # Calculate the performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    r2 = r2_score(y_test, y_pred)
    print(f'R^2 score: {r2}')

    print(f'Predicted price for user input: {linear_model.predict(user_input)}')

    return mse, r2, linear_model

def linearRegressionNoPkg(data, user_input):
    X = data.drop(['price'], axis=1)
    y = data['price']

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    user_input_scaled = scaler.fit_transform(user_input)

    # Adding the intercept term (column of ones) to the features
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    user_input_b = np.c_[np.ones((user_input_scaled.shape[0], 1)), user_input_scaled]

    # Normal Equation: θ = (X^T X)^(-1) X^T y
    theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

    # Predictions
    y_pred = X_test_b.dot(theta_best)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Linear Regression without sklearn:')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R^2 score: {r2}')

    # Predicting the price for user input
    user_pred = user_input_b.dot(theta_best)
    print(f'Predicted price for user input: {user_pred[0]}')

    return r2, mse


# print()
# print("LINEAR REGRESSION USING PACKAGES")
# linearRegressionPkg(dataset, user_input_df)
#
# print()
# print("LINEAR REGRESSION WITHOUT PACKAGES")
# linearRegressionNoPkg(dataset, user_input_df)