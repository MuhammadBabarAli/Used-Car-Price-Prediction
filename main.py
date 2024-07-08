import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from FeatureEncoder import featureEncoder, featureEncoderUserInput

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('pakwheels_used_car_data_v02.csv')

# Handling missing values
dataset = dataset[['make', 'model', 'year', 'transmission', 'fuel', 'city', 'price', 'mileage']]
dataset = dataset.dropna()

# Encoding categorical features
categorical_features = ['make', 'model', 'transmission', 'fuel', 'city']
dataset, encoding_maps = featureEncoder(categorical_features, dataset)

# Splitting the dataset into features and target variable
X = dataset.drop(['price'], axis=1)
y = dataset['price']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


@app.route('/')
def home():
    with open('models/encoding_maps.pkl', 'rb') as f:
        encoding_maps = pickle.load(f)

    return render_template('home.html',
                           make_input='',
                           model_input='',
                           year_input='',
                           mileage_input='',
                           transmission_input='',
                           fuel_input='',
                           city_input='',
                           encoding_maps=encoding_maps,
                           prediction='')


@app.route('/', methods=['GET', 'POST'])
def home_post():
    make_input = request.form['make']
    model_input = request.form['model']
    year_input = float(request.form['year'])
    mileage_input = float(request.form['mileage'])
    transmission_input = request.form['transmission']
    fuel_input = request.form['fuel']
    city_input = request.form['city']
    user_input_df = {
        'make': make_input,
        'model': model_input,
        'year': year_input,
        'transmission': transmission_input,
        'fuel': fuel_input,
        'city': city_input,
        'mileage': mileage_input
    }

    with open('models/linear_reg_pkg.pkl', 'rb') as f:
        linear_reg_pkg = pickle.load(f)

    with open('models/dec_tree_reg_pkg.pkl', 'rb') as f:
        dec_tree_reg_pkg = pickle.load(f)

    with open('models/knn_reg_pkg.pkl', 'rb') as f:
        knn_reg_pkg = pickle.load(f)

    with open('models/encoding_maps.pkl', 'rb') as f:
        encoding_maps = pickle.load(f)

    user_input_df = featureEncoderUserInput(encoding_maps, user_input_df)
    user_input_df = pd.DataFrame([user_input_df])
    prediction = dec_tree_reg_pkg.predict(user_input_df)

    return render_template('home.html',
                           make_input=make_input,
                           model_input=model_input,
                           year_input=year_input,
                           mileage_input=mileage_input,
                           transmission_input=transmission_input,
                           fuel_input=fuel_input,
                           city_input=city_input,
                           encoding_maps=encoding_maps,
                           prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
