"""
Predict Customer Churn

author: Mohammad Khan
Date: 28 May, 2023
"""

# import libraries
import os
import sys
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

# constants
EDA_IMAGE_SAVE_FOLDER = 'images/eda/'
RESULTS_IMAGE_SAVE_FOLDER = 'images/results/'
DATA_PTH = 'data/bank_data.csv'
MODELS_SAVE_FOLDER = 'models/'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        data_frame: pandas dataframe
    '''
    try:
        # read data
        data_frame = pd.read_csv(pth)

        # create y values
        data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return data_frame

    except FileNotFoundError as err:
        print('file not found! ', err)
        return pd.DataFrame()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''

    # churn distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(EDA_IMAGE_SAVE_FOLDER, 'churn_distribution.png'))
    plt.close()

    # customer age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(
        os.path.join(
            EDA_IMAGE_SAVE_FOLDER,
            'customer_age_distribution.png'))
    plt.close()

    # marital status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        os.path.join(
            EDA_IMAGE_SAVE_FOLDER,
            'marital_status_distribution.png'))
    plt.close()

    # Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(
        os.path.join(
            EDA_IMAGE_SAVE_FOLDER,
            'total_transaction_distribution.png'))
    plt.close()

    # correlation
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_SAVE_FOLDER, 'heatmap.png'))
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_list = []
        category_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            category_list.append(category_groups.loc[val])

        target_category_name = category + "_" + response
        df[target_category_name] = category_list

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
        df: pandas dataframe
        response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    # create y
    data_y = df[response]

    # cat_columns = [
    #     'Gender',
    #     'Education_Level',
    #     'Marital_Status',
    #     'Income_Category',
    #     'Card_Category'
    # ]

    # quant_columns = [
    #     'Customer_Age',
    #     'Dependent_count',
    #     'Months_on_book',
    #     'Total_Relationship_Count',
    #     'Months_Inactive_12_mon',
    #     'Contacts_Count_12_mon',
    #     'Credit_Limit',
    #     'Total_Revolving_Bal',
    #     'Avg_Open_To_Buy',
    #     'Total_Amt_Chng_Q4_Q1',
    #     'Total_Trans_Amt',
    #     'Total_Trans_Ct',
    #     'Total_Ct_Chng_Q4_Q1',
    #     'Avg_Utilization_Ratio'
    # ]

    cat_columns = []
    for index, dtype in df.dtypes.items():
        if dtype not in ['float64', 'int64']:
            cat_columns.append(index)

    cat_columns.remove('Attrition_Flag')

    df = encoder_helper(df, cat_columns, response=response)

    data_X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    data_X[keep_cols] = df[keep_cols]

    # This cell may take up to 15-20 minutes to run
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    def model_score_save(model_name,
                         y_train,
                         y_test,
                         y_train_preds,
                         y_test_preds,
                         save_file_name):
        plt.rc('figure', figsize=(5, 5))
        # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old
        # approach
        plt.text(0.01, 1.25, str(model_name + ' Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(model_name + ' Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_IMAGE_SAVE_FOLDER, save_file_name))
        plt.close()

    # random forrest model score save
    model_score_save(
        'Random Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'rf_results.png')

    # logistic regression model score save
    model_score_save(
        'Logistic Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'logistic_results.png')

    # plt.rc('figure', figsize=(5, 5))
    # # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    # plt.text(0.01, 1.25, str('Random Forest Train'), {
    #          'fontsize': 10}, fontproperties='monospace')
    # plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
    #          'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # plt.text(0.01, 0.6, str('Random Forest Test'), {
    #          'fontsize': 10}, fontproperties='monospace')
    # plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
    #          'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # plt.axis('off')
    # plt.savefig(os.path.join(RESULTS_IMAGE_SAVE_FOLDER, 'rf_results.png'))

    # #
    # plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 1.25, str('Logistic Regression Train'),
    #          {'fontsize': 10}, fontproperties='monospace')
    # plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
    #          'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # plt.text(0.01, 0.6, str('Logistic Regression Test'), {
    #          'fontsize': 10}, fontproperties='monospace')
    # plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
    #          'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    # plt.axis('off')
    # plt.savefig(
    #     os.path.join(
    #         RESULTS_IMAGE_SAVE_FOLDER,
    #         'logistic_regression_report_.png'
    #     )
    # )


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, 'feature_importance.png'))
    plt.close()

    # calculate feature impact
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_pth, 'feature_impact.png')
    )
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # train models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # store model scores
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # ROC curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    # plt.show()
    plt.savefig(
        os.path.join(
            RESULTS_IMAGE_SAVE_FOLDER,
            'roc_curve_result.png'))
    plt.close()

    # save best models
    joblib.dump(cv_rfc.best_estimator_, MODELS_SAVE_FOLDER + 'rfc_model.pkl')
    joblib.dump(lrc, MODELS_SAVE_FOLDER + 'logistic_model.pkl')

    # feature importance
    feature_importance_plot(cv_rfc, X_train, RESULTS_IMAGE_SAVE_FOLDER)


if __name__ == "__main__":

    # import data
    print('Importing data')
    data = import_data(DATA_PTH)
    if data.empty:
        sys.exit(-1)
    print('Importing data Complete')

    print('Performing EDA')
    perform_eda(data)
    print('Performing EDA Complete')

    # train test split
    print('Perfroming Feature Engineering')
    _X_train, _X_test, _y_train, _y_test = perform_feature_engineering(
        data, 'Churn')
    print('Perfroming Feature Engineering Complete')

    # train and store model results
    print('Training Models')
    train_models(_X_train, _X_test, _y_train, _y_test)
    print('Training Models Complete')
