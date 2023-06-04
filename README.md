# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, the goal is to implement my learnings from **Clean Code Principles** course to identify credit card customers that are most likely to churn. This project follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).  

## Files and data description

File structure of this project is as below:  
```
.
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Given: Contains the code to be refactored
├── churn_library.py     # functions are defined to predict churn
├── churn_script_logging_and_tests.py # tests and logs codes are here
├── conftest.py          # pytest fixtures are all scripted here for using in test purpose
├── pytest.ini           # pytest configuration to save the logs with logging package
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
├── models               # Store models
├── sequencediagram.jpeg # has sequence of function calls in shurn_library.py file
├── requirements.txt     # python packages required to run this application in docker image.
└── Dockerfile           # Docker file to create docker image for this application
```  

## Running Files
How do you run your files? What should happen when you run your files?

### Create Docker image
```
git clone **TODO**
cd **TODO**
docker build -t mdk/customer-churn:v1.0 .
```

### Run application and tests

Run a docker container first using this command: 
```
docker run -it --gpus all --rm -p 8888:8888 -v "$(pwd)":/app mdk/customer-churn:v1.0
```

Run the applicaton:
```
python3 churn_library.py
```
Run the tests:
```
python3 churn_script_logging_and_tests.py
```


### Important Notes
1. running pytest on training model will take a lot of time. To avoid the test to run training every time, we added a decorator to skip the test for training model. If training model test is required then in churn_script_logging_and_tests.py file the developer has to comment the decorator before the function.   
2. constants.py has the constants but in this version of the software, we did not use it in other source codes. In future releases, this will be integrated with the application source codes.  

