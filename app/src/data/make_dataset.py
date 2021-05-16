import pandas as pd
from ..features.feature_engineering import feature_engineering
from app import cos, init_cols


def make_dataset(data, model_info, cols_to_remove, model_type='RandomForest'):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           data (List):  Lista con la observación llegada por request.
           model_info (dict):  Información del modelo en producción.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame. Dataset a inferir.
    """

    print('---> Getting data')
    print(data)
    print(init_cols)
    data_df = get_raw_data_from_request(data)
    print(data_df)
    print('---> Transforming data')
    data_df = transform_data(data_df, model_info, cols_to_remove)
    
    return data_df.copy()


def get_raw_data_from_request(data):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           data (List):  Lista con la observación llegada por request.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    return pd.DataFrame(data, columns=init_cols)


def transform_data(data_df, model_info, cols_to_remove):
    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
            data_df (DataFrame):  Dataset de entrada.
            model_info (dict):  Información del modelo en producción.
            cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame. Dataset transformado.
    """

    print('------> Removing unnecessary columns')
    print(cols_to_remove)
    print(data_df)
    data_df = remove_unwanted_columns(data_df, cols_to_remove)
    print(data_df)

    #Establezco como indice la columna 'index'
    data_df.set_index('index', inplace=True) 

    # creando dummies originales
    # print('------> Encoding data')
    #print('---------> Getting encoded columns from cos')
    #enc_key = model_info['objects']['encoders']+'.pkl'
    # obteniendo las columnas presentes en el entrenamiento desde COS
    #enc_cols = cos.get_object_in_cos(enc_key)
    
    # agregando las columnas dummies faltantes en los datos de entrada
    #data_df = data_df.reindex(columns=enc_cols, fill_value=0)

    return data_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)





