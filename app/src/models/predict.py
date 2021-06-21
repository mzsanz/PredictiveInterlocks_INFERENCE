from ..data.make_dataset import make_dataset
from app import cos, client
from cloudant.query import Query


def predict_pipeline(data, model_info_db_name='predictive-interlocks-model'):

    """
        Function that implements the full inference pipeline of the model.

        Args:
            path (str):  Path to data.

        Kwargs:
            model_info_db_name (str):  database to store model info.

        Returns:
            list. List with the predictions.
    """

    # Load of the training model configuration
    model_config = load_model_config(model_info_db_name)['model_config']
    print(model_config)
    # columns to remove
    cols_to_remove = model_config['cols_to_remove']
    print(cols_to_remove)
    # obtaining the info from the model in production
    model_info = get_best_model_info(model_info_db_name)
    print(model_info)
    # Loading and transforming the input data
    data_df = make_dataset(data, model_info, cols_to_remove)

    # Downloading the model object
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)
    #model = load_model("model_1621097730.pkl")
    print(model)
    # doing the inference with the input data
    return model.predict(data_df).tolist()


def load_model(name, bucket_name='uem-models-mzs'):
    """
         Function to load the model in IBM COS

         Args:
             name (str):  Name of the object in COS to load.

         Kwargs:
             bucket_name (str):  bucket of IBM COS to be used.

        Returns:
            obj. Downloaded object.
     """
    return cos.get_object_in_cos(name, bucket_name)


def get_best_model_info(db_name):
    """
         Function to load the model info from IBM Cloudant

         Args:
             db_name (str):  database to use.

         Kwargs:
             bucket_name (str):  IBM COS bucket to be used.

        Returns:
            dict. Model info.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]


def load_model_config(db_name):
    """
        Function to load the model info from IBM Cloudant.

        Args:
            db_name (str):  database.

        Returns:
            dict. Document with the model config.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]
