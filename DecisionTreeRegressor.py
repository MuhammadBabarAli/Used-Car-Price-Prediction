from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from FeatureEncoder import featureEncoder, featureEncoderUserInput
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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



def decisionTreeRegressorPkg(data, user_input):
    # Define features (X) and target (y)
    X = data.drop(columns=['price'])
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Decision Tree Regressor
    dt_regressor = DecisionTreeRegressor(random_state=42, max_depth=13)

    # Fit the model
    dt_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = dt_regressor.predict(X_test)

    # Calculate the performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    print(f'R² Score: {r2}')

    print(dt_regressor.predict(user_input))

    return mse, r2, dt_regressor


def decisionTreeRegressorNoPkg(data, user_input, max_depth=12, min_samples_split=2, min_impurity_decrease=1e-7):
    # Custom decision tree regressor functions
    def calculate_mse(y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def find_best_split(X, y):
        best_mse = float('inf')
        best_split = None
        n_features = X.shape[1]

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = X[:, feature] > value

                y_left, y_right = y[left_mask], y[right_mask]
                mse_left, mse_right = calculate_mse(y_left), calculate_mse(y_right)
                mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, value)

        return best_split

    def build_tree(X, y, depth=0):
        # Stopping conditions
        if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < min_samples_split:
            return np.mean(y)

        feature, value = find_best_split(X, y)
        if feature is None:
            return np.mean(y)

        left_mask = X[:, feature] <= value
        right_mask = X[:, feature] > value

        left_tree = build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = build_tree(X[right_mask], y[right_mask], depth + 1)

        # Pruning: Stop splitting if no significant reduction in impurity
        current_impurity = calculate_mse(y)
        left_impurity = calculate_mse(y[left_mask])
        right_impurity = calculate_mse(y[right_mask])
        weighted_impurity = (len(y[left_mask]) * left_impurity + len(y[right_mask]) * right_impurity) / len(y)

        if (current_impurity - weighted_impurity) < min_impurity_decrease:
            return np.mean(y)

        return (feature, value, left_tree, right_tree)

    def predict_tree(tree, x):
        if not isinstance(tree, tuple):
            return tree

        feature, value, left_tree, right_tree = tree
        if x[feature] <= value:
            return predict_tree(left_tree, x)
        else:
            return predict_tree(right_tree, x)

    # Define features (X) and target (y)
    X = data.drop(columns=['price'])
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the tree with the improved stopping and pruning criteria
    tree = build_tree(X_train.values, y_train.values)

    # Predict on the test set
    y_pred_custom = [predict_tree(tree, x) for x in X_test.values]

    # Calculate the performance
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    rmse = np.sqrt(mse_custom)
    r2 = r2_score(y_test, y_pred_custom)

    print(f'Mean Squared Error: {mse_custom}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R² Score: {r2}')

    # Predict for user input
    user_predictions = [predict_tree(tree, x) for x in user_input.values]
    print('User Predictions:', user_predictions)

    return r2, mse_custom


# print()
# print("DECISION TREE REGRESSION USING PACKAGES")
# dec_tree_reg_pkg_model = decisionTreeRegressorPkg(dataset, user_input_df)
#
# print()
# print("DECISION TREE REGRESSION WITHOUT PACKAGES")
# decisionTreeRegressorNoPkg(dataset, user_input_df)
