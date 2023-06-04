"""
Customer Churn prediction test script

author: Mohammad Khan
Date: 31 May, 2023
"""


import os
import logging
import pandas as pd
import joblib
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    # format='%(name)s - %(levelname)s - %(message)s'
    # https://stackoverflow.com/questions/10973362/python-logging-function-name-file-name-line-number-using-a-single-file
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
)


def test_import(import_data, request):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    request.config.cache.set('cache_df', dataframe.to_json())


def test_eda(perform_eda, eda_outputs, temp_folder, request):
    '''
    test perform eda function
    '''
    # load the output of import_data() and check df size to verify
    try:
        dataframe = request.config.cache.get('cache_df', None)
        dataframe = pd.read_json(dataframe)
        assert dataframe.shape[0] > 0
        logging.info("Testing perform_eda: cached df found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("dataframe: \n {}".format(dataframe.head().to_string()))

    except BaseException as err:
        logging.error("Testing perform_eda: cached df is not found")
        raise err

    # test eda
    try:
        expected_files = eda_outputs
        logging.info('Testing perform_eda: start')
        cls.EDA_IMAGE_SAVE_FOLDER = temp_folder
        logging.info(
            'Testing perform_eda: running the function and saving in temp folder: {}'.format(
                cls.EDA_IMAGE_SAVE_FOLDER))
        
        # call the fixture
        perform_eda(dataframe)

        # checking files are created or not.
        generated_files = os.listdir(cls.EDA_IMAGE_SAVE_FOLDER)

        logging.info('generated files: {}'.format(generated_files))
        logging.info('expected files: {}'.format(expected_files))

        assert len(generated_files) == len(expected_files)
        assert set(generated_files) == set(expected_files)
        # assert sorted(generated_files) == sorted(expected_files)
        logging.info('Testing perform_eda: all the eda images are created.')

        # clean up
        logging.info('Testing perform_eda: cleanup eda images starts')
        try:
            if os.path.exists(cls.EDA_IMAGE_SAVE_FOLDER):
                # cleanup
                for file_name in os.listdir(cls.EDA_IMAGE_SAVE_FOLDER):
                    # construct full file path
                    file = os.path.join(cls.EDA_IMAGE_SAVE_FOLDER, file_name)
                    if os.path.isfile(file):
                        logging.info('Deleting file:{}'.format(file))
                        os.remove(file)
            logging.info('Testing perform_eda: cleanup eda images ends')

        except BaseException:
            logging.error('TEMP_FOLDER could not be cleaned up!!!')

    except Exception as err:
        logging.error("Testing perform_eda: generated images missing")
        raise err

    logging.info('Testing perform_eda: SUCCESS')


def test_encoder_helper(encoder_helper, request):
    '''
    test encoder helper
    '''
    # load the output of import_data() and check df size to verify
    try:
        df = request.config.cache.get('cache_df', None)
        df = pd.read_json(df)
        assert df.shape[0] > 0
        logging.info("Testing encoder_helper: cached df found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("df: \n {}".format(df.head().to_string()))

    except Exception as err_load:
        logging.error("Testing encoder_helper: cached df is not found")
        raise err_load

    try:
        logging.info('Testing encoder_helper: start')

        # test encoder_helper
        cat_columns = []
        for index, dtype in df.dtypes.items():
            if dtype not in ['float64', 'int64']:
                cat_columns.append(index)

        cat_columns.remove('Attrition_Flag')

        encoded_df = encoder_helper(df, cat_columns, response='Churn')

        assert encoded_df.shape[0] > 0
        assert encoded_df.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The df doesn't have the right encoded columns")
        raise err

    try:
        # test encoder_helper
        cat_columns = []
        for index, dtype in df.dtypes.items():
            if dtype not in ['float64', 'int64']:
                cat_columns.append(index)

        cat_columns.remove('Attrition_Flag')

        encoded_df = encoder_helper(df, cat_columns, response='Churn')

        # expected columns vs encoded_df columns
        for cat_column in cat_columns:
            expected_column = cat_column + "_Churn"
            assert expected_column in encoded_df.columns

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The df doesn't have the right encoded columns")
        raise err

    logging.info("Testing encoder_helper: SUCCESS")

    # request.config.cache.set('cache_encoded_df', encoded_df.to_json())


def test_perform_feature_engineering(perform_feature_engineering, request):
    '''
    test perform_feature_engineering
    '''
    # load the output of import_data() and check df size to verify
    try:
        df = request.config.cache.get('cache_df', None)
        df = pd.read_json(df)
        assert df.shape[0] > 0
        logging.info("Testing perform_feature_engineering: cached df found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("df: \n {}".format(df.head().to_string()))

    except Exception as err_load:
        logging.error(
            "Testing perform_feature_engineering: cached df is not found")
        raise err_load

    try:

        logging.info('Testing perform_feature_engineering: start')

        _X_train, _X_test, _y_train, _y_test = perform_feature_engineering(
            df, 'Churn')

        assert _X_train.shape[0] > 0
        assert _X_train.shape[1] > 0

        assert _X_test.shape[0] > 0
        assert _X_test.shape[1] > 0

        assert len(_X_train) == len(_y_train)
        assert len(_X_test) == len(_y_test)

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: wrong feature engineering")
        raise err

    # request push data for train models test
    request.config.cache.set('cache_x_train', _X_train.to_json())
    request.config.cache.set('cache_y_train', _y_train.to_json())

    request.config.cache.set('cache_x_test', _X_test.to_json())
    request.config.cache.set('cache_y_test', _y_test.to_json())

    logging.info("y_train: \n {}".format(_y_train.head().to_string()))
    logging.info('y_train type: {}'.format(type(_y_train)))
    logging.info("y_test: \n {}".format(_y_test.head().to_string()))

    logging.info("Testing perform_feature_engineering: SUCCESS")


@pytest.mark.skip(reason="model training takes a long time. Not worth testing every time.")
def test_train_models(train_models, request):
    '''
    test train_models
    '''
    # load the output of import_data() and check df size to verify
    try:
        x_train = request.config.cache.get('cache_x_train', None)
        x_train = pd.read_json(x_train)
        assert x_train.shape[0] > 0
        logging.info("Testing train_models: cached x_train found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("x_train: \n {}".format(x_train.head().to_string()))

    except Exception as err_load:
        logging.error("Testing train_models: cached x_train is not found")
        raise err_load

    try:
        y_train = request.config.cache.get('cache_y_train', None)
        # https://stackoverflow.com/questions/22965985/pandas-series-to-json-and-back
        y_train = pd.read_json(y_train, typ='series', orient='records')
        # assert y_train.shape[0] > 0
        logging.info("Testing train_models: cached y_train found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("y_train: \n {}".format(y_train.head().to_string()))

    except Exception as err_load:
        logging.error("Testing train_models: cached y_train is not found")
        raise err_load

    try:
        x_test = request.config.cache.get('cache_x_test', None)
        x_test = pd.read_json(x_test)
        assert x_test.shape[0] > 0
        logging.info("Testing train_models: cached x_test found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("x_test: \n {}".format(x_test.head().to_string()))

    except Exception as err_load:
        logging.error("Testing train_models: cached x_test is not found")
        raise err_load

    try:
        y_test = request.config.cache.get('cache_y_test', None)
        # https://stackoverflow.com/questions/22965985/pandas-series-to-json-and-back
        y_test = pd.read_json(y_test, typ='series', orient='records')
        assert y_test.shape[0] > 0
        logging.info("Testing train_models: cached y_test found: ")
        # https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
        logging.info("y_test: \n {}".format(y_test.head().to_string()))

    except Exception as err_load:
        logging.error("Testing train_models: cached y_test is not found")
        raise err_load

    try:
        train_models(x_train, x_test, y_train, y_test)
    except Exception as err_train:
        logging.error(
            'Testing train_models: training function did not run properly')
        raise err_train

    try:
        joblib.load("models/rfc_model.pkl")
        joblib.load("models/logistic_model.pkl")
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models: The weight files path is wrong/ files were not present.")
        raise err


if __name__ == "__main__":
    # pytest.main(args=['-p no:logging', os.path.abspath(__file__)])
    pass
