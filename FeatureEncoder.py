def featureEncoder(categorical_features, dataset):
    unique_options = {}

    # Get unique values for all features
    for feature in categorical_features:
        unique_values = dataset[feature].unique()
        unique_options[feature] = unique_values.tolist()

    # Encoding categorical features
    def label_encode(series):
        unique_values = series.unique()
        value_to_number = {value: number for number, value in enumerate(unique_values)}
        return series.map(value_to_number), value_to_number

    encoded_features = {}
    encoding_maps = {}

    # Create feature encoding map
    for feature in categorical_features:
        encoded_column, value_to_number = label_encode(dataset[feature])
        dataset[feature] = encoded_column
        encoding_maps[feature] = value_to_number
        encoded_features[feature] = {
            "value_to_number": value_to_number,
        }

    # # Display the unique options and their encodings
    # print("\nEncoding Maps for each Categorical Feature:\n")
    # for feature in categorical_features:
    #     print(f"{feature} Encoding:")
    #     print("Value to Number Mapping:", encoding_maps[feature][0])
    #     print("Number to Value Mapping:", encoding_maps[feature][1])
    #     print("\n")

    return dataset, encoding_maps

def featureEncoderUserInput(encoding_maps, user_input_df):
    for k in user_input_df.keys():
        if k == 'year' or k == 'mileage':
            pass
        else:
            val = encoding_maps[k][user_input_df[k]]
            user_input_df[k] = val

    return user_input_df