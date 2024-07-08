import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from FeatureEncoder import featureEncoder, featureEncoderUserInput
import numpy as np
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

def knnRegressorPkg(data, user_input):
    # Splitting the dataset into features and target variable
    X = data.drop('price', axis=1)
    y = data['price']

    # Scaling the featuresgit 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Applying the KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=10)  # Initial number of neighbors
    knn.fit(X_train, y_train)

    # Predicting the test set
    y_pred = knn.predict(X_test)

    # Evaluating the model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    # Hyperparameter tuning using GridSearchCV
    param_grid = {'n_neighbors': np.arange(1, 25)}  # Test K values from 1 to 24
    grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    # Best parameters
    print(f'Best parameters: {grid.best_params_}')

    # Refit the model with the best parameters
    best_knn = grid.best_estimator_
    best_knn.fit(X_train, y_train)
    y_pred_best = best_knn.predict(X_test)

    # Evaluate the refitted model
    best_mse = mean_squared_error(y_test, y_pred_best)
    best_rmse = np.sqrt(best_mse)
    print("After hyper parameter tuning")
    print(f'Best Mean Squared Error: {best_mse}')
    print(f'Best Root Mean Squared Error: {best_rmse}')

    # Calculate R²
    best_r2 = r2_score(y_test, y_pred_best)
    print(f'Best R² Score: {best_r2}')

    print(best_knn.predict(user_input))

    return mse, best_r2, best_knn


def knnRegressorNoPkg(data, user_input):
    # Ensure data types are correct
    data = data.apply(pd.to_numeric, errors='coerce')
    user_input = np.array(user_input, dtype=float)

    # Splitting the dataset into features and target variable
    X = data.drop('price', axis=1).values
    y = data['price'].values

    # Manually scaling the features
    def normalize_features(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # Prevent division by zero for features with zero variance
        return (X - X_mean) / X_std, X_mean, X_std

    X_scaled, X_mean, X_std = normalize_features(X)

    # Normalize user input using the same scaler
    def normalize_user_input(user_input, X_mean, X_std):
        return (user_input - X_mean) / X_std

    user_input_scaled = normalize_user_input(user_input, X_mean, X_std)

    # Splitting the data into training and testing sets
    def train_test_split_custom(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    X_train, X_test, y_train, y_test = train_test_split_custom(X_scaled, y, test_size=0.2)

    # KNN Regressor manual
    def euclidean_distance(X_train, test_point):
        return np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))

    def knn_predict(X_train, y_train, X_test, k):
        y_pred = []
        for test_point in X_test:
            distances = euclidean_distance(X_train, test_point)
            k_nearest_indices = np.argsort(distances)[:k]
            k_nearest_values = y_train[k_nearest_indices]
            y_pred.append(np.mean(k_nearest_values))
        return np.array(y_pred)

    # Predicting the test set with K=5
    k = 5
    y_pred = knn_predict(X_train, y_train, X_test, k)

    # Predicting the user input
    user_pred = knn_predict(X_train, y_train, user_input_scaled, k)

    # Evaluating the model performance
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def r2_score(y_true, y_pred):
        total_variance = np.var(y_true, ddof=1)
        explained_variance = np.var(y_true - y_pred, ddof=1)
        return 1 - (explained_variance / total_variance)

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R^2 Score: {r2}')

    print(f'Predicted price for user input: {user_pred}')

    return r2, mse


# print("KNN REGRESSOR USING PACKAGES")
# knn_reg_pkg_model = knnRegressorPkg(dataset, user_input_df)
#
# print()
# print("KNN REGRESSOR WITHOUT PACKAGES")
# knnRegressorNoPkg(dataset, user_input_df)