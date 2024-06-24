"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
import joblib


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## Missing in outcome avaiable
    missing = df["outcome_available"] == 0
    
    df = df.drop(df[missing].index, axis = "rows")
    
    # Selecting variables for modelling
    keepcols = [
        "nomem_encr",  # ID variable required for predictions,
        "age_bg",         # age
        "gender_bg",
        "migration_background_bg",
        "oplcat_2020",
        "woning_2020",
        "cf20m024",
        "cf20m128",
        "cf20m454"
      
    ] 

    # Keeping data with variables selected
    df = df[keepcols]

    # delete all rows with missing values

    X_isna = df.isna().any(axis=1)

    df = df.drop(df[X_isna].index)

    # process the data
    # select the row of IDs
    df_ID = df["nomem_encr"]
    
    # process other variables
    df['age_bg'] = df['age_bg'].astype(str).astype(float).astype(int)
    df['gender_bg'] = df['gender_bg'].astype(str)
    df['migration_background_bg'] = df['migration_background_bg'].astype(str)
    df['oplcat_2020'] = df['oplcat_2020'].astype(str) # education
    df['woning_2020'] = df['woning_2020'].astype(str) # rent a house or not
    df['cf20m024'] = df['cf20m024'].astype(str) # have a partner or not
    df['cf20m128'] = df['cf20m128'].astype(str) # fertility intention
    df['cf20m454'] = df['cf20m454'].astype(str) # have a child or not


    numerical_columns = ['age_bg']

    categorical_columns = ['gender_bg', 'migration_background_bg', "oplcat_2020", "woning_2020", "cf20m024", "cf20m128", "cf20m454"]

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output = False)
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)
]).set_output(transform = "pandas")

    df = preprocessor.fit_transform(df)

    df = pd.merge(df, df_ID, left_index = True, right_index=True)

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
