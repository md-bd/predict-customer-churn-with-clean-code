"""
Customer Churn prediction test script helper conftest

author: Mohammad Khan
Date: 31 May, 2023
"""

import os
import logging
import pytest
import churn_library as cls


@pytest.fixture
def import_data():
    return cls.import_data


@pytest.fixture
def perform_eda():
    return cls.perform_eda


@pytest.fixture
def encoder_helper():
    return cls.encoder_helper


@pytest.fixture
def perform_feature_engineering():
    return cls.perform_feature_engineering


@pytest.fixture
def train_models():
    return cls.train_models


@pytest.fixture
def eda_outputs():
    gen_files = [
        'churn_distribution.png', 
        'marital_status_distribution.png',
        'customer_age_distribution.png',
        'total_transaction_distribution.png',
        'heatmap.png',
    ]
    return gen_files


@pytest.fixture
def temp_folder():
    
    TEMP_FOLDER = 'images/temp'
    
    try:
        if not os.path.exists(TEMP_FOLDER):
            os.makedirs(TEMP_FOLDER)
    except:
        logging.error('TEMP_FOLDER could not be created!!!')
    
    return TEMP_FOLDER

