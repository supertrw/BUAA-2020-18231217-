import random
import math
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split as holdout
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import sklearn.metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test_sample.csv')
test_output = test.copy()

label_sex = LabelEncoder()
label_smoker = LabelEncoder()
label_region = LabelEncoder()
train['sex'] = label_sex.fit_transform(train['sex'])
train['smoker'] = label_smoker.fit_transform(train['smoker'])
train['region'] = label_region.fit_transform(train['region'])

test['sex'] = label_sex.fit_transform(test['sex'])
test['smoker'] = label_smoker.fit_transform(test['smoker'])
test['region'] = label_region.fit_transform(test['region'])

train['sk_bmi'] = '0'
test['sk_bmi'] = '0'
train['agefix'] = '0'
test['agefix'] = '0'

# group_df = train.groupby(pd.Grouper(key='age')).mean()
# #     print(group_df)
# group_df = group_df.sort_index()
# #     print(group_df)
# group_df.plot(y = ['charges'],kind = 'bar')
# plt.savefig('../picture/age.png')
# plt.show()

# for i in range(len(train)):
#     if train['smoker'][i] == 0:
#         train['sk_bmi'][i] = 0
#     elif train['bmi'][i] < 30:
#         train['sk_bmi'][i] = 1
#     else:
#         train['sk_bmi'][i] = 2
#
#     if train['age'][i] >= 18 and train['age'][i] <= 32:
#         train['agefix'] = 1
#     elif train['age'][i] >= 33 and train['age'][i] <= 42:
#         train['agefix'] = 2
#     elif train['age'][i] >= 43 and train['age'][i] <= 58:
#         train['agefix'] = 3
#     else:
#         train['agefix'] = 4

# for i in range(len(test)):
#     if test['smoker'][i] == 0:
#         test['sk_bmi'][i] = 0
#     elif test['bmi'][i] < 30:
#         test['sk_bmi'][i] = 1
#     else:
#         test['sk_bmi'][i] = 2
#
#     if test['age'][i] >= 18 and test['age'][i] <= 32:
#         test['agefix'] = 1
#     elif test['age'][i] >= 33 and test['age'][i] <= 42:
#         test['agefix'] = 2
#     elif test['age'][i] >= 43 and test['age'][i] <= 58:
#         test['agefix'] = 3
#     else:
#         train['agefix'] = 4
feature = ['sex', 'smoker', 'region', 'age', 'bmi', 'children']

X = train[feature]
# sc = StandardScaler()
# X = sc.fit_transform(X)
Y = train['charges']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def Visualization():
    predict = {'RMSE': [2673, 2894, 2943, 3594, 3879],
            'Model': ['RandomForest','GBDT','DecisionTree','SVR','LinerRegression']
            }
    predict = pd.DataFrame(predict)
    f, axe = plt.subplots(1, 1, figsize=(18, 12))
    sns.barplot(x='Model', y='RMSE', data=predict, ax=axe)
    axe.set_xlabel('Model', size=21)
    axe.set_ylabel('RMSE', size=21)
    plt.savefig('../picture/模型.png')
    plt.show()

    # sns.scatterplot(x=train['bmi'], y=train['charges'], hue=train['smoker'])
    # plt.savefig('../picture/bmi_smoker.png')
    # plt.show()

    # sns.distplot(train["age"], bins=10)
    # plt.savefig('../picture/age.png')
    # plt.show()
    #
    # sns.distplot(train["bmi"], bins=10)
    # plt.savefig('../picture/bmi.png')
    # plt.show()
    #
    # sns.violinplot(x="sex", y="charges", data=train)
    # plt.savefig('../picture/sex.png')
    # plt.show()
    #
    # sns.violinplot(x="smoker", y="charges", data=train)
    # plt.savefig('../picture/smoker.png')
    # plt.show()
    #
    # sns.violinplot(x="region", y="charges", data=train)
    # plt.savefig('../picture/region.png')
    # plt.show()
    #
    # sns.violinplot(x="children", y="charges", data=train)
    # plt.savefig('../picture/children.png')
    # plt.show()
    #
    # ins_smoker = train[train["smoker"] == 1]
    # corrs = ins_smoker[['age', 'bmi', 'children', 'region', 'charges']].corr()
    # sns.heatmap(corrs, linewidths=0.5, annot=True, center=0, cmap="YlGnBu")
    # plt.title('Smokers')
    # plt.savefig('../picture/Smoker.png')
    # plt.show()
    #
    # ins_non_smoker = train[train["smoker"] == 0]
    # corrs = ins_non_smoker[['age', 'bmi', 'children', 'region', 'charges']].corr()
    # sns.heatmap(corrs, linewidths=0.5, annot=True, center=0, cmap="YlGnBu")
    # plt.title('non_Smokers')
    # plt.savefig('../picture/non_smoker.png')
    # plt.show()
    #
    # corrMatrix = train.corr()
    # sns.set(font_scale=1.10)
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
    #             square=True, annot=True, cmap='RdPu')
    # plt.savefig('../picture/热力图.png')
    # plt.show()
    #
    # sns.pairplot(train, hue = "smoker")
    # plt.savefig('../picture/关系图_smoker.png')
    # plt.show()
    #
    # sns.pairplot(train, hue="age")
    # plt.savefig('../picture/关系图_age.png')
    # plt.show()
    #
    # sns.pairplot(train, hue="sex")
    # plt.savefig('../picture/关系图_sex.png')
    # plt.show()
    #
    # sns.pairplot(train, hue="bmi")
    # plt.savefig('../picture/关系图_bmi.png')
    # plt.show()
    #
    # sns.pairplot(train, hue="children")
    # plt.savefig('../picture/关系图_children.png')
    # plt.show()
    #
    # sns.pairplot(train, hue="region")
    # plt.savefig('../picture/关系图_region.png')
    # plt.show()

def RMSE(Y_true,Y_pred):
    return np.sum((Y_true-Y_pred)**2)**0.5/len(Y_true)

class DecisionTree_Myself(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.left_child = None
        self.right_child = None

    def predict_value(self, X):
        if self.leaf_value:
            return self.leaf_value
        elif X[self.split_feature] <= self.split_value:
            return self.left_child.predict_value(X)
        else:
            return self.right_child.predict_value(X)

class RF_Myself(object):
    def __init__(self, n_estimators=100, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, mRadio=0.8, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.mRadio = mRadio
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances = dict()

    def fit(self, X, y):
        if self.random_state:
            random.seed(self.random_state)
        if self.max_features == "sqrt":
            self.max_features = int(len(X.columns) ** 0.5)
        elif self.max_features == "log2":
            self.max_features = int(math.log(len(X.columns)))
        else:
            self.max_features = len(X.columns)
        y = y.to_frame(name='charges')
        for stage in range(self.n_estimators):
            print("正在建立第{}棵树".format(stage + 1))
            X_stage = X.sample(n=int(self.mRadio * len(X)), replace=True,
                               random_state=stage).reset_index(drop=True)
            col_tree = random.sample(X.columns.tolist(), self.max_features)
            X_stage = X_stage.loc[:, col_tree]
            y_stage = y.sample(n=int(self.mRadio * len(X)), replace=True,
                               random_state=stage).reset_index(drop=True)
            tree = self.build_tree(X_stage, y_stage, depth=0)
            self.trees[stage] = tree

    def build_tree(self, X, y, depth):
        if len(y['charges'].unique()) <= 1 or len(X) <= self.min_samples_split or depth >= self.max_depth:
            tree = DecisionTree_Myself()
            tree.leaf_value = y['charges'].mean()
            return tree
        best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(X, y)
        left_X = X[X[best_split_feature] <= best_split_value]
        left_y = y[X[best_split_feature] <= best_split_value]
        right_X = X[X[best_split_feature] > best_split_value]
        right_y = y[X[best_split_feature] > best_split_value]
        tree = DecisionTree_Myself()
        if len(left_X) <= self.min_samples_leaf or len(right_X) <= self.min_samples_leaf:
            tree.leaf_value = y['charges'].mean()
            return tree
        else:
            self.feature_importances[best_split_feature] = self.feature_importances.get(best_split_feature, 0) + 1
            tree.split_feature = best_split_feature
            tree.split_value = best_split_value
            tree.left_child = self.build_tree(left_X, left_y, depth + 1)
            tree.right_child = self.build_tree(right_X, right_y, depth + 1)
            return tree

    def choose_best_feature(self, X, y):
        best_split_gain = float("inf")
        best_split_feature = None
        best_split_value = None
        for feature in X.columns:
            if len(X[feature].unique()) <= 100:
                unique_values = sorted(X[feature].unique().tolist())
            else:
                unique_values = np.unique([np.percentile(X[feature], x) for x in np.linspace(0, 100, 100)])
            for split_value in unique_values:
                left_y = y[X[feature] <= split_value]
                right_y = y[X[feature] > split_value]
                split_gain = 0
                for iy in [left_y['charges'], right_y['charges']]:
                    mean = iy.mean()
                    for dt in iy:
                        split_gain += (dt - mean) ** 2
                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    def predict(self, X):
        ans = []
        for col, row in X.iterrows():
            pred_list = []
            for i, tree in self.trees.items():
                pred_list.append(tree.predict_value(row))
            ans.append(sum(pred_list) / len(pred_list))
        return np.array(ans)

def RandomForest_byMyself():

    clf = RF_Myself(n_estimators=10,
                    max_depth=6,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    mRadio=0.8,
                    random_state=2020)
    # clf.fit(train[feature], train['charges'])
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print('evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    predY = clf.predict(test)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_RF_byMyself.csv', index=False)

def LinearRegression_byMyself_Model():
    Y_train = y_train.values.reshape((-1, 1))
    Y_test = y_test.values.reshape((-1, 1))
    n_sample = X_train.shape[0]
    n_feature = X_train.shape[1]
    W = np.random.randn(n_feature).reshape((n_feature, 1))
    b = 1
    Y_hat = np.dot(X_train, W) + b

    max_iter = 5000
    alpha = 0.05
    for i in range(max_iter):
        Y_hat = np.dot(X_train, W) + b
        dW = 2 * X_train.T.dot(Y_hat - Y_train) / n_sample
        db = 2 * np.sum(Y_hat - Y_train) / n_sample
        W = W - alpha * dW
        b = b - alpha * db
    Y_pred_train = np.dot(X_train, W) + b
    Y_pred_test = np.dot(X_test, W) + b
    print("Train RMSE: ", RMSE(Y_train, Y_pred_train))
    print("Test  RMSE: ", RMSE(Y_test, Y_pred_test))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = np.dot(testX, W) + b
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_LinearRegression_byMyself.csv', index=False)


def RandomForest_Model():
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = regressor.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_RandomForest.csv', index=False)

def RandomForest2_Model():
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf
                   }

    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=2, random_state=42)
    rf_random.fit(X_train, y_train)
    best_estimator = rf_random.best_estimator_
    print("最佳参数:")
    print(best_estimator)


    y_train_pred = rf_random.predict(X_train)
    y_test_pred = rf_random.predict(X_test)

    print('RandomForestRegressor evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = rf_random.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_RandomForest2.csv', index=False)

def RandomForest3_Model():
    regressor_rf = RandomForestRegressor()
    parameters = {"n_estimators": [1200],
                  "max_features": ["auto"],
                  "max_depth": [50],
                  "min_samples_split": [7],
                  "min_samples_leaf": [10],
                  "bootstrap": [True],
                  "criterion": ["mse"],
                  "random_state": [42]}

    regressor_rf = GridSearchCV(estimator=regressor_rf,
                                param_grid=parameters,
                                cv=10,
                                verbose = 4)
    regressor_rf.fit(X_train, y_train)

    y_train_pred = regressor_rf.predict(X_train)
    y_test_pred = regressor_rf.predict(X_test)

    print(regressor_rf.best_estimator_)
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = regressor_rf.predict(testX)
    test_output['charges'] = predY

    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_RandomForest3.csv', index=False)

def LinearRegression_Model():
    x = train.drop(['charges'], axis=1)
    y = train['charges']
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
    Lin_reg = LinearRegression()
    Lin_reg.fit(x_train, y_train)

    y_train_pred = Lin_reg.predict(x_train)
    y_test_pred = Lin_reg.predict(x_test)

    print('evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))
    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = Lin_reg.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_LinearRegression.csv', index=False)

def LassoRegression_Model():
    x = train.drop(['charges'], axis=1)
    y = train['charges']
    x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
    Lasso_reg = Lasso()
    Lasso_reg.fit(x_train, y_train)

    y_train_pred = Lasso_reg.predict(x_train)
    y_test_pred = Lasso_reg.predict(x_test)

    print('evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))
    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = Lasso_reg.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_LassoRegression.csv', index=False)

def GBDT_Model_RandomizedSearchCV():
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=50)]
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 4, 5, 6, 7, 8, 9, 10]
    min_samples_leaf = [1, 2, 4, 6, 8,10]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf
                   }

    gbdt = GradientBoostingRegressor()
    gbdt_random = RandomizedSearchCV(estimator=gbdt, param_distributions=random_grid,
                                   n_iter=100, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=2, random_state=42)
    gbdt_random.fit(X_train, y_train)

    best_estimator = gbdt_random.best_estimator_
    print("最佳参数:")
    print(best_estimator)

    y_train_pred = gbdt_random.predict(X_train)
    y_test_pred = gbdt_random.predict(X_test)

    print('GBDT evaluating result:')
    print("Train 绝对误差: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train 均方根误差: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test 绝对误差: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test 均方根误差: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = gbdt_random.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_GBDT_Aoh.csv', index=False)

def GBDT_Model_GridSearchCV():
    param_gbdt = {'n_estimators': list(range(500, 600, 10)),
                  'learning_rate': [0.03, 0.04, 0.05, 0.06],
                  'max_depth': [4, 5, 6, 7, 8, 9,10],
                  'min_samples_split': [2, 3, 5, 6, 10, 20, 50, 100, 300, 400, 500, 600],
                  'min_samples_leaf': [1, 2, 4, 5, 8, 10, 20, 30, 50]
                  }
    gbdt_search = GridSearchCV(estimator=GradientBoostingRegressor(max_features='sqrt', subsample=0.8,random_state=75),
                                param_grid=param_gbdt, scoring='neg_mean_squared_error', iid=False, cv=5)

    gbdt_search.fit(X_train, y_train)
    print(gbdt_search.best_params_)
    print(gbdt_search.best_score_)
    # gbdt_search = GradientBoostingRegressor()
    # # gbdt_search.fit(X_train, y_train)
    # gbdt_search.fit(X, Y)
    y_train_pred = gbdt_search.predict(X_train)
    y_test_pred = gbdt_search.predict(X_test)
    # gbdt = GradientBoostingRegressor()
    # gbdt = GradientBoostingRegressor(learning_rate = 0.06, n_estimators = 580,
    #                                  min_samples_split=500, min_samples_leaf=30,
    #                                  max_depth=6, max_features='sqrt', subsample=0.8,
    #                                  random_state=75)
    # gbdt.fit(X_train, y_train)

    # y_train_pred = gbdt.predict(X_train)
    # y_test_pred = gbdt.predict(X_test)

    print('evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = gbdt_search.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_GBDT.csv', index=False)

def SVR_Model():
    parameters = {'kernel': ['rbf', 'sigmoid'],
                  'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
                  'tol': [0.0001],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    regressor_svr = SVR()
    regressor_svr = GridSearchCV(estimator=regressor_svr,
                                 param_grid=parameters,
                                 cv=10,
                                 verbose=2,
                                 iid=True)
    regressor_svr = regressor_svr.fit(X_train, y_train)
    print("最佳参数:")
    print(regressor_svr.best_estimator_)

    y_train_pred = regressor_svr.predict(X_train)
    y_test_pred = regressor_svr.predict(X_test)

    print('RandomForestRegressor evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = regressor_svr.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_SVR.csv', index=False)

def DecisionTree_Model():
    regressor_dt = DecisionTreeRegressor(random_state=2021)
    parameters = [{"max_depth": np.arange(1, 21),
                   "min_samples_leaf": [1, 5, 10, 20, 50, 100],
                   "min_samples_split": np.arange(2, 11),
                   "criterion": ["mse"],
                   "random_state": [42]}
                  ]
    regressor_dt = GridSearchCV(estimator=regressor_dt,
                                param_grid=parameters,
                                cv=10,
                                verbose=4,
                                iid=False)
    regressor_dt = regressor_dt.fit(X_train, y_train)
    print("最佳参数:")
    print(regressor_dt.best_estimator_)

    y_train_pred = regressor_dt.predict(X_train)
    y_test_pred = regressor_dt.predict(X_test)

    print('RandomForestRegressor evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = regressor_dt.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_DecisionTree.csv', index=False)

def KNeighborsRegressor_Model():
    select = SelectKBest(k='all')
    knn = KNeighborsRegressor()
    pipeline = make_pipeline(select, knn)
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print('evaluating result:')
    print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
    print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
    print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = pipeline.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_KNeighborsRegressor.csv', index=False)

def rf_gbdt_dt_fusion():
    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    gbdt_search = GradientBoostingRegressor()
    gbdt_search.fit(X, Y)
    Y_gbdt = gbdt_search.predict(testX)

    rf = RandomForestRegressor(max_depth=90,
                               min_samples_leaf=4,
                               n_estimators=400
                              )
    rf.fit(X_train, y_train)
    Y_rf = rf.predict(testX)


    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=2020)
    dt.fit(X_train, y_train)
    Y_dt = rf.predict(testX)

    Y_ans = []
    for i in range(len(Y_gbdt)):
        Y_ans.append((Y_dt[i] * 1 + Y_gbdt[i] * 2 + Y_rf[i] * 3) / 6)
    test_output['charges'] = Y_ans
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_rf_gbdt_dt_fusion.csv', index=False)

def model_fusion1():
    rfans = pd.read_csv('../data/submission_RandomForest.csv')
    gbdtans = pd.read_csv('../data/submission_GBDT.csv')
    svrans = pd.read_csv('../data/submission_SVR.csv')
    lrans = pd.read_csv('../data/submission_LinearRegression.csv')
    dtans = pd.read_csv('../data/submission_DecisionTree.csv')

    rfans = rfans['charges']
    gbdtans = gbdtans['charges']
    svrans = svrans['charges']
    lrans = lrans['charges']
    dtans = dtans['charges']

    Y_ans = []
    for i in range(len(rfans)):
        Y_ans.append(rfans[i] + gbdtans[i] + svrans[i] + lrans[i] + dtans[i])
    test_output['charges'] = Y_ans
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_model_fusion.csv', index=False)

def model_fusion2():
    rfans = pd.read_csv('../data/submission_RandomForest.csv')
    gbdtans = pd.read_csv('../data/submission_GBDT.csv')
    svrans = pd.read_csv('../data/submission_SVR.csv')
    lrans = pd.read_csv('../data/submission_LinearRegression.csv')
    dtans = pd.read_csv('../data/submission_DecisionTree.csv')

    rfans = rfans['charges']
    gbdtans = gbdtans['charges']
    svrans = svrans['charges']
    lrans = lrans['charges']
    dtans = dtans['charges']

    Y_ans = []
    for i in range(len(rfans)):
        Y_ans.append((rfans[i] * 5 + gbdtans[i] * 4 + svrans[i] * 3 + lrans[i] * 2 + dtans[i] * 1) / 15)
    test_output['charges'] = Y_ans
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_model_fusion.csv', index=False)

def model_fusion3():
    rfans = pd.read_csv('../data/submission_RandomForest.csv')
    gbdtans = pd.read_csv('../data/submission_GBDT.csv')
    svrans = pd.read_csv('../data/submission_SVR.csv')
    lrans = pd.read_csv('../data/submission_LinearRegression.csv')
    dtans = pd.read_csv('../data/submission_DecisionTree.csv')

    rfans = rfans['charges']
    gbdtans = gbdtans['charges']
    svrans = svrans['charges']
    lrans = lrans['charges']
    dtans = dtans['charges']

    Y_ans = []
    for i in range(len(rfans)):
        k = ((rfans[i] * 5 + gbdtans[i] * 4 + svrans[i] * 3 + lrans[i] * 2 + dtans[i] * 1) / 15)
        if abs(rfans[i] - k) > rfans[i] * 0.25:
            k = (gbdtans[i] * 4 + svrans[i] * 3 + lrans[i] * 2 + dtans[i] * 1) / 10
        elif abs(gbdtans[i] - k) > gbdtans[i] * 0.25:
            k = (rfans[i] * 5 + svrans[i] * 3 + lrans[i] * 2 + dtans[i] * 1) / 11
        elif abs(svrans[i] - k) > svrans[i] * 0.25:
            k = (rfans[i] * 5 + gbdtans[i] * 4 + lrans[i] * 2 + dtans[i] * 1) / 12
        elif abs(lrans[i] - k) > lrans[i] * 0.25:
            k = (rfans[i] * 5 + gbdtans[i] * 4 + svrans[i] * 3 + dtans[i] * 1) / 13
        elif abs(dtans[i] - k) > dtans[i] * 0.25:
            k = (rfans[i] * 5 + gbdtans[i] * 4 + svrans[i] * 3 + lrans[i] * 2) / 14
        Y_ans.append(k)
    test_output['charges'] = Y_ans
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_model_fusion.csv', index=False)

def model_stacking():
    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    gbdt = GradientBoostingRegressor()
    gbdt.fit(X, Y)
    Y_gbdt = gbdt.predict(X)

    # rf = RandomForestRegressor()
    # rf.fit(X, Y)
    # Y_rf = rf.predict(X)

    svr = SVR()
    svr.fit(X, Y)
    Y_svr = svr.predict(X)

    dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=2020)
    dt.fit(X,Y)
    Y_dt = dt.predict(X)

    lr = LinearRegression()
    lr.fit(X,Y)
    Y_lr = lr.predict(X)

    newinput = []
    for i in range(len(Y_gbdt)):
        newi = [Y_gbdt[i], Y_lr[i], Y_dt[i], Y_svr[i]]
        newinput.append(newi)

    rf = RandomForestRegressor(max_depth=90,
                               min_samples_leaf=4,
                               n_estimators=400)
    rf.fit(newinput, Y)

    O_gbdt = gbdt.predict(testX)
    O_lr = lr.predict(testX)
    O_dt = dt.predict(testX)
    O_svr = svr.predict(testX)

    newoutput = []
    for i in range(len(O_gbdt)):
        newo = [O_gbdt[i], O_lr[i], O_dt[i], O_svr[i]]
        newoutput.append(newo)

    Y_ans = rf.predict(newoutput)
    test_output['charges'] = Y_ans
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_model_stacking.csv', index=False)

def rf_Model_one_out():
    regressor = RandomForestRegressor(max_depth=90,
                                      min_samples_leaf=4,
                                      n_estimators=400
                                      )
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X, Y):
        print(train_index)
        print(test_index)
        regressor.fit(X[train_index], Y[train_index])
        Y_train_pred = regressor.predict(X[train_index])
        Y_test_pred = regressor.predict(X[test_index])
        print("Train RMSE: ", RMSE(Y[train_index], Y_train_pred))
        print("Test RMSE: ", RMSE(Y[test_index], Y_test_pred))
        print()

    testX = test[feature]
    sc = StandardScaler()
    testX = sc.fit_transform(testX)

    predY = regressor.predict(testX)
    test_output['charges'] = predY
    submission = pd.DataFrame.from_dict(test_output)
    submission.to_csv('../data/submission_rf_K_Flods.csv', index=False)

######################################################
'''选择合适的模型进行训练即可'''
# Visualization()                 # 可视化函数
RandomForest_byMyself()         # 手写实现的随机森林模型
# LinearRegression_byMyself_Model()   # 手写实现的线性回归
# RandomForest_Model()            # 手动调参的随机森林
# RandomForest2_Model()           # RandomizedSearchCV调参的模型
# RandomForest3_Model()           # GridSearchCV调参的模型
# LinearRegression_Model()        # 线性回归
# LassoRegression_Model()         # Lasso回归
# KNeighborsRegressor_Model()     # K近邻
# SVR_Model()                     # SVR模型
# DecisionTree_Model()            # 决策树模型
# GBDT_Model_RandomizedSearchCV() # RandomizedSearchCV调参的模型
# GBDT_Model_GridSearchCV()       # GridSearchCV调参的模型
# rf_gbdt_dt_fusion()             # 模型融合1
# model_fusion1()                 # 直接平均融合
# model_fusion2()                 # 加权平均融合
# model_fusion3()                 # 优化后的加权平均
# model_stacking()                # 堆叠法融合