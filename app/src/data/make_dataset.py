import pandas as pd
from ..features.feature_engineering import feature_engineering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from app import cos, init_cols


def make_dataset(data, model_info, cols_to_remove):

    """
        Function that performs data transformations.

        Args:
           data (List):  List with the new requested observation.
           model_info (dict):  Info from the model in production.

        Kwargs:

        Returns:
           DataFrame. Dataset to infere.
    """

    print('---> Getting data')
    print(data)
    print(init_cols)
    data_df = get_raw_data_from_request(data)
    print(data_df)
    print('---> Transforming data and making Feature Engineering')
    data_df = transform_data(data_df, model_info, cols_to_remove)
    print('---> Inputing and scaling')
    data_df = pre_train_data_prep(data_df, model_info)
    
    return data_df.copy()


def get_raw_data_from_request(data):

    """
        Function to obtain new observations

        Args:
           data (List):  List with the requested observation.

        Returns:
           DataFrame. Dataset with the input data.
    """
    return pd.DataFrame(data, columns=init_cols)


def transform_data(data_df, model_info, cols_to_remove):
    """
        Function that transforms the input data and makes feature engineering.

        Args:
            data_df (DataFrame):  Data input.
            model_info (dict):  Info. from the model in production.
            cols_to_remove (list): Cols to remove.

        Returns:
           DataFrame. Transformed data.
    """

    # Removing senseless data related to 'impossible' beam destinations
    print('------> Removing senseless data')
    data_df = remove_senseless(data_df)
    print(data_df)

    #Adding new predictors (Feature Engineering)
    print('------> Adding new predictors')
    data_df = add_predictors(data_df)
    print(data_df)

    #Removing rows with BM 'No beam'
    print('------> Removing data with BM=NoBeam')
    data_df = remove_rows_BM_zero(data_df)
    print(data_df)

    # Removing BM column
    print('------> Removing BM columns')
    print(cols_to_remove)
    data_df = remove_unwanted_columns(data_df, cols_to_remove)
    print(data_df)

    #Establezco como indice la columna 'index'
    #data_df.set_index('index', inplace=True) 

    return data_df.copy()


def remove_senseless(df):
    """
        Function to remove imposible beam destinations

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    index_names_BD5 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 0) & (df['BD_0'] == 1)].index
    index_names_BD6 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 1) & (df['BD_0'] == 0)].index
    index_names_BD7 = df[ (df['BD_2'] == 1) & (df['BD_1'] == 1) & (df['BD_0'] == 1)].index
    df.drop(index_names_BD5, inplace = True)
    df.drop(index_names_BD6, inplace = True)
    df.drop(index_names_BD7, inplace = True)
    
    return (df)

def add_predictors(df):
    """
        Function to add new predictors (Feature Engineering)

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    df['Section_1'] = ((df['GV1'] == 1) & (df['GV2'] == 1) & (df['VBP1']==1) & (df['VBP2']==1)).astype(int)
    df['Section_2'] = ((df['GV3'] == 1) & (df['GV4'] == 1) & (df['VBP3']==1) & (df['VBP4']==1)).astype(int)
    df['Section_3'] = ((df['GV5'] == 1) & (df['VBP5']==1)).astype(int) 
    df['Section_4'] = ((df['GV6'] == 1) & (df['GV7'] == 1) & (df['VBP6']==1) & (df['VBP7']==1)).astype(int) 
    df['BtT'] = (df['Section_1'] & df['Section_2'] & df['Section_3'] & df['Section_4']).astype(int)
    return (df)


def remove_rows_BM_zero(df):
    """
        Function to remove data rows with BM zero

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    index_names = df[ (df['BM'] == 0) ].index
    df.drop(index_names, inplace = True)
    return (df)


def remove_unwanted_columns(df, cols_to_remove):
    """
        Function to remove unnneded columns

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)


def pre_train_data_prep(data_df, model_info):
    """
        Function that makes the last transformations on the dataset NULL imputing and scaling
        Args:
           data_df (DataFrame):  dataset.
           
        Returns:
           DataFrame. transformed dataset.
    """

    # NULL imputing
    print('------> Getting imputer from cos')
    imputer_key = model_info['objects']['imputer']+'.pkl'
    data_df = input_missing_values(data_df, imputer_key)

    # Scaling
    print('------> Getting scaler from cos')
    scaler_key = model_info['objects']['scaler']+'.pkl'
    data_df = scale_data(data_df, scaler_key)

    return data_df.copy()

def input_missing_values(data_df, key):
    """
        Function for NULLs imputing
        Args:
           data_df (DataFrame):  dataset.

        Returns:
           DataFrame. transformed dataset.
    """
    
    print('------> Inputing missing values')
    # obtain the SimpleImputer object from  COS
    imputer = cos.get_object_in_cos(key)
    data_df = pd.DataFrame(imputer.transform(data_df), columns=data_df.columns)
    print(data_df)

    return data_df.copy()

def scale_data(data_df, key):
    """
        Function to scale variables
        Args:
           data_df (DataFrame):  dataset.
           Returns:
           DataFrame. dataset transformed.
    """

    print('------> Scaling values')
    # obtain the Scaled object from  COS
    scaler = cos.get_object_in_cos(key)
    data_df = pd.DataFrame(scaler.transform(data_df), columns=data_df.columns)

    return data_df.copy()


