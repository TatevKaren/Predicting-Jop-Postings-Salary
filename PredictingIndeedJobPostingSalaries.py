import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

print('--------------------------------------------------------------------------------')
print("Load Visualize and Clean Data")
print('\n --------------------------------------------------------------------------------')

class load_visualize_Data:
    def __init__(self, train_features, train_response, test_features, pred_response):
        '''loading training features '''
        self.train_features = train_features
        '''loading training response data'''
        self.train_response = train_response
        '''loading test features'''
        self.test_features = test_features
        # the response variable name
        self.pred_response = pred_response
        '''checking the data and getting statistics'''
        self.process_validate_data()
        '''Distribution Plots'''
        self._distplot()
        '''Boxplots'''
        self._boxplot()


    def process_validate_data(self):
        '''data exploration and validation process'''
        # load and store training data to dataFrame
        self._create_training_df()
        # load and store test data to dataFrame
        self._create_test_df()
        # get information about the features: categorical and numerical features
        self._get_features_info()
        # get training data statistics
        self._get_trainDf_statistics()
        # get test data statistics
        self._get_testDf_statistics()
        # checking for missing data points
        self._check_missing_data(self.trainDf)
        # checking the number of repeatitions/duplicates
        self._check_duplicates()
        # getting all columns containing negative or 0 values which is something we don't expect
        self._get_invalid_data(self.trainDf, pred_response)

    def _create_training_df(self):
        '''loading and preparing training data '''
        train_featuresDf = pd.read_csv(train_features, header=0)
        train_responseDf = pd.read_csv(train_response, header=0)
        # combining the train features and response
        self.trainDf = train_featuresDf.merge(train_responseDf, on=None, how="inner")

    def _create_test_df(self):
        '''loading and preparing training data'''
        self.testDf = pd.read_csv(test_features, header=0)

    def _get_features_info(self):
        '''getting categorical and numerical features'''
        self.features_cat = self._get_cat_features(self.trainDf)
        self.features_num = self._get_num_features(self.trainDf)

    def _get_cat_features(self, df):
        '''finding  categorical columns in Dataframe'''
        self.features_cat = df.select_dtypes(include=['O']).columns.tolist()
        print('List of Categorical Features: {}'.format(self.features_cat))
        return (self.features_cat)

    def _get_num_features(self, df):
        '''finding numerical columns in Dataframe'''
        self.features_num = df.select_dtypes(exclude=['O']).columns.tolist()
        print('List of Numerical Features: {}'.format(self.features_num))
        return (self.features_num)

    def _get_trainDf_statistics(self):
        print('Training Data Statistics')
        self._get_statistics(self.trainDf)

    def _get_testDf_statistics(self):
        print('Test Data Statistics')
        self._get_statistics(self.testDf)

    def _get_statistics(self, df):
        print('\n-----------------------------------------------------------------------------')
        print('\n  Dataframe Information: \n')
        print('n{}'.format(df.info()))
        print('--------------------------------------------------------------------------------')
        print('\n Dataframe Size [#rows, #cols]- {}'.format(df.shape))
        print('\n Numerical Features Statistics: \n \n{}'.format(df.describe()))
        print('\n Categorical Features Stats: \n \n{}'.format(df.describe(include='O')))
        print('--------------------------------------------------------------------------------')

    def _check_missing_data(self, df):
        '''Checking and finding  null or na values in Dataframe'''
        num_missingval = np.sum(df.isna().sum()) + np.sum(df.isnull().sum())
        if num_missingval == 0:
            print('\n\n : There are no missing data points in the data')
        else:
            print('Features or columns that contain missing values\n\n{}'.format(df.isnull().sum()))

    def _check_duplicates(self):
        '''Checking for duplicates'''
        print('\n : There are {} duplicate values in Train Data'.format(self.trainDf.duplicated().sum()))
        # though we found 5 repetitions in salary feature, this is not a duplicate since multiple jobs can have the same salary
        print('\n : There are {} duplicate values in Test Data'.format(self.testDf.duplicated().sum()))

    def _get_invalid_data(self, df, cols):
        '''Finding and flagging invalid values'''
        for i in [cols]:
            # we don't expect any of the values to be equal to 0 so we will identify anything <= 0
            inv_counts = np.sum(df[i] <= 0)
            if inv_counts > 0:
                self.invalid_data = True
                print('\n :There are {} duplicates in {} column'.format(inv_counts, i))

    def _distplot(self):
        '''Creates Distribution Plots for Numeric Features'''
        fig = plt.figure(figsize=(14, 10))
        for index, col in enumerate(self.features_num):
            fig.add_subplot(len(self.features_num), len(self.features_num), index + 1)
            n, x, _ = plt.hist(self.trainDf[col], bins=20, color='yellow', edgecolor='black', linewidth=0.5)
            bin_centers = 0.5 * (x[1:] + x[:-1])
            plt.plot(bin_centers, n, color='darkgreen', linewidth=2)
            plt.title('Distribution Plot for Numeric Features')
            plt.xlabel(str(col))
            plt.tight_layout()

    def _boxplot(self):
        '''Creates BoxPlots for Categorical and Numeric Features'''
        df = self.trainDf.copy()
        fig = plt.figure(figsize=(14, 9))
        for index, col in enumerate(self.features_cat):
            if len(self.trainDf[col].unique()) < 10:
                df[col + '_mean'] = df.groupby(col)[self.pred_response].transform('mean')
                fig.add_subplot(4, 2, index + 1)
                sns.boxplot(x=col, y=self.pred_response, data=df.sort_values(col + '_mean'))
                plt.title('Salaries vs {}'.format(col), fontsize=12)
                plt.tight_layout()
                plt.xticks(rotation=45)
        for index, col in enumerate(self.features_num):
            fig.add_subplot(len(self.features_num), len(self.features_num), index + 1)
            sns.boxplot(self.trainDf[col], color='yellow')
            plt.tight_layout()


pred_response = 'salary'
train_features = 'train_features.csv'
train_response = 'train_salaries.csv'
test_features = 'test_features.csv'
data = load_visualize_Data(train_features, train_response, test_features, pred_response)

# Using Interquantile Range Rule (IQR) to get the lower and upper limits for the response varaible
stat = data.trainDf.salary.describe()
InterQuantRange = stat['75%'] - stat['25%']
upper_lev = stat['75%'] + 1.5 * InterQuantRange
lower_lev = stat['25%'] - 1.5 * InterQuantRange
print(data.trainDf[data.trainDf.salary < lower_lev])
print('\n The upper and lower salary levels for possible outliers are {} and {}.'.format(upper_lev, lower_lev))
print('these values seem unreasonable and can be assumed to be outliers because salary here is equal to 0 while they are not even internships or unpaid type of work')

print(data.trainDf[data.trainDf.salary > upper_lev])
print('\n these values seem reasonable because these high salaries correspodng to job requiring experience and these are executive positions')

# removing these outliers from the data
data.trainDf = data.trainDf[data.trainDf.salary > lower_lev].reset_index()
print('\n --------------------------------------------------------------------------------')
print("Data Preprocessing")
print('\n-----------------------------------------------------------------------------')


class dataPrepocessing:
    def __init__(self, data):
        self.data = data
        self.features_cat = ['companyId', 'jobType', 'degree', 'major', 'industry']
        self.pred_response = data.pred_response
        # we will store here the labels we get from labels encoder
        self.labels = {}
        self.invalid_data = data.invalid_data
        self.clean_encode_df()

    def clean_encode_df(self):
        '''Cleaning Data From Invalid Data points/Errors'''
        '''Since salaries equal to 0 (lower limit we found) don't make sense, /n we assume they are outliers and we will fag and remove them'''
        if self.invalid_data:
            print('Number of data points before removing invalid rows:- {}'.format(data.trainDf.shape[0]))
            data.trainDf = data.trainDf[data.trainDf['salary'] > 0]
            print('Number of data points after removing invalid rows with zero salary:- {}'.format(data.trainDf.shape[0]))

        ''' Encoding the categorical labels in training data'''
        trainDf = self.encode_cat_features(data.trainDf, self.features_cat)
        # since this is unique per observations it doesn't make sense to encode it
        # but we still need the jobIds, therefore we will use it as the index
        self.trainDf = trainDf.set_index("jobId").drop("index",1)


        ''' Encoding the categorical labels in test data'''
        testDf = self.encode_cat_features(data.testDf, self.features_cat, test_data=True)
        self.testDf = testDf.set_index("jobId")


    def encode_cat_features(self, df, features, test_data=False):
        '''encoding the labels in categorical features'''
        if not test_data:
            for feature in features:
                l_encoder = LabelEncoder()
                l_encoder.fit(df[feature])
                self.labels[feature] = l_encoder
                df[feature] = l_encoder.transform(df[feature])
        else:
            # encoding for test data
            for feature, l_encoder in self.labels.items():
                df[feature] = l_encoder.transform(df[feature])
        return (df)

data_cleaned = dataPrepocessing(data)

print('\n --------------------------------------------------------------------------------')
print("Training and Predictions")
print('--------------------------------------------------------------------------------')

class SalaryPredictingModel:
    def __init__(self, data, models):
        '''training multiple ML models for predicting salaries'''
        self.data = data
        self.models = models
        self.mse = {}
        self.rmse = {}
        self.mae = {}
        self.best_model = None
        self.predictions = None
        self.pred_response = data.pred_response
        self.train_response = data.trainDf[data.pred_response]
        self.train_features = data.trainDf.drop(data.pred_response, axis=1)
        self.testDf = data.testDf
        self.train_models()

    def train_models(self):
        self.KFold_CV_model()
        self.get_best_model()

    print('Training Process with K-Fold Cross Validation')
    print('--------------------------------------------------------------------------------')
    ''' we will use K-fold CV for estimating average test error rate'''

    # defaults is K=5 in K-fold
    def KFold_CV_model(self):
        for model_output in self.models:
            print("Training model" + str(model_output) +" and calculating CV MSE, CV MAE, CV RMSE")
            scores_mse = cross_val_score(model_output, self.train_features, self.train_response, cv=5,
                                         scoring='neg_mean_squared_error')
            scores_mae = cross_val_score(model_output, self.train_features, self.train_response, cv=5,
                                         scoring='neg_mean_absolute_error')
            scores_rmse = cross_val_score(model_output, self.train_features, self.train_response, cv=5,
                                          scoring='neg_root_mean_squared_error')
            self.mse[model_output] = -1.0 * np.mean(scores_mse)
            self.mae[model_output] = -1.0 * np.mean(scores_mae)
            self.rmse[model_output] = -1.0 * np.mean(scores_rmse)

    # picking the model with the least CV RMSE, then fitting and predicting that model
    def get_best_model(self):
        '''Selecting the best model with RMSE, fitting the model train data'''
        self.best_model = min(self.rmse, key=self.rmse.get)
        self.get_model_performance()

    def get_model_performance(self):
        print("Model Performance")
        for key, item in self.rmse.items():
            print('\n Score of the model {} :-'.format(key))
            print('\n RMSE - {}'.format(item))
        print('\n Best model with smallest RMSE\n\n {} :-'.format(self.best_model))
        print('\n RMSE - {}'.format(self.rmse[self.best_model]))
        print('\nTraining the Best Model.....')
        self.best_model.fit(self.train_features, self.train_response)
        print('\n Getting Feature Importance')
        self._plot_feature_importance()
        print('\nPrediction for test data with Best Model')
        self.testDf[self.pred_response] = self.best_model.predict(self.testDf)
        print('-------------------------- Best Model Performance ------------------------')
        print(self.testDf[self.pred_response])

    def _plot_feature_importance(self):
        '''Printing the feature importance used to train the model'''
        print('\n Feature Importance Calculation')
        features = self.train_features.columns.to_list()
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(7, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)),
                 importances[indices], color='r', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
        

    @staticmethod
    def saving_best_model(model_file, model):
        ''' Saving the best model to a file'''
        print('\n Saving Best Model to file')
        pickle.dump(model, open(model_file, 'wb'))
    print('Saving the Predictions to CSV file')
    def saving_results(self, sub_file, df):
        print('\n Saving job Salary predictions in testDf to a CSV file')
        print('--------------------------------------------------------------------------------')
        self.testDf[self.pred_response].to_csv(sub_file, index=True, header = 0)

    @staticmethod
    def hyperparameter_tuning(estimator, param_grid, n_iter=5, scoring='neg_root_mean_absolute_error', cv=5, n_jobs=-2,
                              refit=False):
        ''' Finding Optimal hyper-parameters used in models'''
        rs_cv = RandomizedSearchCV(estimator=estimator, param_distribution=param_grid,
                                   n_iter=n_iter,
                                   cv=cv,
                                   n_jobs=n_jobs,
                                   refit=refit)
        rs_cv.fit(train_features, train_response)
        return (rs_cv.best_params_)


pred_response = 'salary'
train_features = 'train_features.csv'
train_response = 'train_salaries.csv'
test_features = 'test_features.csv'

train_featuresDf = pd.read_csv(train_features, header=0)
train_responseDf = pd.read_csv(train_response, header=0)
trainDf = train_featuresDf.merge(train_responseDf, on=None, how="inner")
testDf = pd.read_csv(test_features, header=0)


LR = linear_model.LinearRegression()
Lasso_LR = linear_model.Lasso()
RForest = RandomForestRegressor(n_estimators=60, n_jobs=4, max_depth=15, min_samples_split=80, max_features=8,verbose=0)
GBM = GradientBoostingRegressor(n_estimators=60, max_depth=7, loss='ls', verbose=0)
xgboost = XGBRegressor(n_estimators=60, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
models = [LR, Lasso_LR, RForest, GBM, xgboost]
model_output = SalaryPredictingModel(data_cleaned, models)
# saving the submission file using the Best model
sub_file = './Test_Salaries.csv'
model_output.saving_results(sub_file,model_output.testDf)
# Saving the Best model to a file.
model_file = './SalaryPrediction_Best_Model.sav'
model_output.saving_best_model(model_file, model_output.best_model)
