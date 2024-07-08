import pandas as pd
from KNNRegressor import knnRegressorPkg, knnRegressorNoPkg
from DecisionTreeRegressor import decisionTreeRegressorPkg, decisionTreeRegressorNoPkg
from LinearRegression import linearRegressionPkg, linearRegressionNoPkg
from FeatureEncoder import featureEncoder, featureEncoderUserInput
import matplotlib.pyplot as plt
import pickle

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

# Handling missing values
dataset = dataset[['make', 'model', 'year', 'transmission', 'fuel', 'city', 'price', 'mileage']]
dataset = dataset.dropna()

# Encoding categorical features
categorical_features = ['make', 'model', 'transmission', 'fuel', 'city']
dataset, encoding_maps = featureEncoder(categorical_features, dataset)
user_input_df = featureEncoderUserInput(encoding_maps, user_input_df)

with open('models/encoding_maps.pkl', 'wb') as f:
    pickle.dump(encoding_maps, f)

user_input_df = pd.DataFrame([user_input_df])

print("KNN REGRESSOR USING PACKAGES")
knn_pkg_mse, knn_pkg_r2, knn_reg_pkg_model = knnRegressorPkg(dataset, user_input_df)
with open('models/knn_reg_pkg.pkl', 'wb') as f:
    pickle.dump(knn_reg_pkg_model,f)

# Slow
print()
print("KNN REGRESSOR WITHOUT PACKAGES")
knn_nopkg_r2, knn_nopkg_mse = knnRegressorNoPkg(dataset, user_input_df)

print()
print("LINEAR REGRESSION USING PACKAGES")
linear_pkg_mse, linear_pkg_r2, linear_reg_pkg_model = linearRegressionPkg(dataset, user_input_df)
with open('models/linear_reg_pkg.pkl', 'wb') as f:
    pickle.dump(linear_reg_pkg_model, f)

print()
print("LINEAR REGRESSION WITHOUT PACKAGES")
linear_nopkg_r2, linear_nopkg_mse = linearRegressionNoPkg(dataset, user_input_df)

print()
print("DECISION TREE REGRESSION USING PACKAGES")
dec_tree_pkg_mse, dec_tree_pkg_r2, dec_tree_reg_pkg_model = decisionTreeRegressorPkg(dataset, user_input_df)
with open('models/dec_tree_reg_pkg.pkl', 'wb') as f:
    pickle.dump(dec_tree_reg_pkg_model,f)

print()
print("DECISION TREE REGRESSION WITHOUT PACKAGES")
dec_tree_nopkg_r2, dec_tree_nopkg_mse = decisionTreeRegressorNoPkg(dataset, user_input_df)


# Plotting the results
models = ['KNN (Pkg)', 'KNN (No Pkg)', 'Linear (Pkg)', 'Linear (No Pkg)', 'Decision Tree (Pkg)', 'Decision Tree (No Pkg)']
mse_scores = [knn_pkg_mse, knn_nopkg_mse, linear_pkg_mse, linear_nopkg_mse, dec_tree_pkg_mse, dec_tree_nopkg_mse]
r2_scores = [knn_pkg_r2, knn_nopkg_r2, linear_pkg_r2, linear_nopkg_r2, dec_tree_pkg_r2, dec_tree_nopkg_r2]

# Plot MSE scores
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison of Regression Models')
plt.xticks(rotation=45)

# Plot R² scores
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color='lightgreen')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('R² Score Comparison of Regression Models')
plt.xticks(rotation=45)

# Show plots
plt.tight_layout()
plt.show()

# Detailed comparison of the models
comparison_df = pd.DataFrame({
    'Model': models,
    'MSE': mse_scores,
    'R² Score': r2_scores
})

print("\nDetailed Comparison of Models:")
print(comparison_df)