import numpy as np
import pandas as pd
import sys
import pickle
import yaml
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
# from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import LearningCurveDisplay


def flatten(xss):
    return [x for xs in xss for x in xs]


def print_log(text):

    # original_stdout = sys.stdout  # save reference
    print(text)
    with open('results/results.txt', 'a') as f:
        # sys.stdout = f  # Change the standard output to the file we created.
        # sys.stdout = original_stdout      # Reset the standard output to its original value
        f.write(f'{text}\n')
        f.write('\n')


class ModelComparator():

    def __init__(self, df_clean, is_regression=False, study_residuals=True):
        self.df_clean = df_clean.reset_index(drop=True)
        self.X = self.df_clean.drop('Y', axis=1)
        self.Y = self.df_clean['Y']

        self.n = df_clean.shape[0]
        self.p = len(self.X.columns)
        self.continuous_features = self.X.select_dtypes(include=['number']).columns
        self.categorical_features = self.X.select_dtypes(include='object').columns

        self.is_regression = is_regression
        self.study_residuals = study_residuals

        self.config = self.load_config("compareModelConfig.yaml")
        # print(self.config.keys())
        self.init_pipeline(
            poly_deg=self.config['pipeline_preprocessing']['poly_deg'],
            spline_knots=self.config['pipeline_preprocessing']['spline_knots'],
            spline_degree=self.config['pipeline_preprocessing']['spline_degree']
            )
        self.init_algo_regression() if is_regression is True else self.init_algo_classification()

    @staticmethod
    def load_config(config_name):
        CONFIG_PATH = "config"
        with open(os.path.join(CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)
            print(".....configuration file loaded.....")
        return config

    def init_pipeline(self, poly_deg, spline_knots, spline_degree):
        """
        Pipeline:
        - Categorical
        - replace nan value par most frequent
        - one hot encode en enlevant une colone
        - Continuous
            - replace nan value par mean
            - standardize

        On lit le fichier JSON de parametrisation des modeles
        """

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        poly_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', PolynomialFeatures(poly_deg)),
            ('scaler', StandardScaler())
        ])

        spline_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', SplineTransformer(n_knots=spline_knots, degree=spline_degree)),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('continuous', continuous_transformer, self.continuous_features),
            ('categorical', categorical_transformer, self.categorical_features)
        ])

        poly_preprocessor = ColumnTransformer(transformers=[
            # on drop les features non polynomiales
            ('continuous', continuous_transformer,
             self.config["pipeline_preprocessing"]['non_interaction_features']),
            ('categorical', categorical_transformer, self.categorical_features),
            ('poly', poly_transformer,
             self.continuous_features.drop(self.config["pipeline_preprocessing"]
                                           ['non_interaction_features'])),
        ])

        poly_spline_preprocessor = ColumnTransformer(transformers=[
            ('continuous', continuous_transformer,
             self.config["pipeline_preprocessing"]['non_interaction_features']),
            ('categorical', categorical_transformer, self.categorical_features),
            ('poly', poly_transformer,
             self.continuous_features.drop(self.config["pipeline_preprocessing"]
                                           ['non_interaction_features'])),
            ('spline', spline_transformer,
             self.continuous_features.drop(self.config["pipeline_preprocessing"]
                                           ['non_interaction_features'])),
        ])

        dict_preprocesseurs = {"simple": preprocessor,
                               "poly": poly_preprocessor,
                               "poly and spline": poly_spline_preprocessor
                               }

        self._preprocesseurs = {f'{prepro}': dict_preprocesseurs[prepro]
                                for prepro in self.config["pipeline_preprocessing"]
                                                         ['preprocesseurs_all_algo']}

        return None

    def init_algo_classification(self):

        # ON CHARGE LES PARAMETRES DU FICHIER DE CONFIG
        params_class_ridge = {
            # 'preprocessor__continuous__scaler' : [StandardScaler(), MinMaxScaler(), Normalizer()],
            'classifier__C': np.logspace(-1, 5, 20)
        }

        params_class_lasso = {
            'classifier__C': np.logspace(-1, 5, 20)
        }

        params_class_elastic = {
            'classifier__C': np.logspace(-1, 5, 20),
            'classifier__l1_ratio': [0.25, 0.5, 0.75]
        }

        params_class_tree = {
            'classifier__min_samples_split': [2, 5]
        }

        params_class_rf = {
            'classifier__n_estimators': [10, 100],
            'classifier__max_depth': [90, 100],
            'classifier__max_features': [2, 3],
        }

        params_class_gb = {
        }

        self.dict_algo = {
                "Logistic Regression": (LogisticRegression(penalty=None, fit_intercept=True), None),
                "Ridge": (LogisticRegression(penalty="l2", fit_intercept=True), params_class_ridge),
                "Lasso": (LogisticRegression(penalty="l1", solver="saga", fit_intercept=True),
                          params_class_lasso),
                "ElsaticNet": (LogisticRegression(penalty="elasticnet", solver="saga",
                               fit_intercept=True), params_class_elastic),
                "Decision Tree": (DecisionTreeClassifier(), params_class_tree),
                "Random Forest": (RandomForestClassifier(),
                                  params_class_rf),
                "KNN": (KNeighborsClassifier(), self.config["params_classif"]['params_knn']),
                "GBoosting": (GradientBoostingClassifier(),
                              params_class_gb)
        }

        self.dict_algo = {
            key: self.dict_algo[key] for key in self.config["input_models"]['classif']
            }

        return None

    def init_algo_regression(self):

        # ON CHARGE LES PARAMETRES DU FICHIER DE CONFIG

        params_reg_linear = {}

        params_reg_ridge = {
            'classifier__alpha': np.logspace(
                self.config["params_reg"]['params_ridge']['alpha_start'],
                self.config["params_reg"]['params_ridge']['alpha_stop'],
                self.config["params_reg"]['params_ridge']['alpha_num'],
            )
        }

        params_reg_lasso = {
            'classifier__alpha': np.logspace(
                self.config["params_reg"]['params_lasso']['alpha_start'],
                self.config["params_reg"]['params_lasso']['alpha_stop'],
                self.config["params_reg"]['params_lasso']['alpha_num'],
            ),
            'classifier__max_iter': self.config["params_reg"]['params_lasso']['max_iter'],
            'classifier__tol': self.config["params_reg"]['params_lasso']['tol'],
        }

        params_reg_elastic = {
            'classifier__alpha': np.logspace(
                self.config["params_reg"]['params_elastic']['alpha_start'],
                self.config["params_reg"]['params_elastic']['alpha_stop'],
                self.config["params_reg"]['params_elastic']['alpha_num'],
            ),
            'classifier__max_iter': self.config["params_reg"]['params_elastic']['max_iter'],
            'classifier__tol': self.config["params_reg"]['params_elastic']['tol'],
        }

        params_reg_tree = {
            'classifier__min_samples_split': self.config["params_reg"]
                                                        ['params_tree']['min_samples_split'],
        }

        params_reg_forest = {
            'classifier__n_estimators': self.config["params_reg"]['params_forest']['n_estimators'],
            'classifier__max_depth': self.config["params_reg"]['params_forest']['max_depth'],
            'classifier__max_features': self.config["params_reg"]['params_forest']['max_features'],
        }

        params_reg_gb = {
            'classifier__n_estimators': self.config["params_reg"]['params_gb']['n_estimators'],
            'classifier__max_depth': self.config["params_reg"]['params_gb']['max_depth'],
            'classifier__learning_rate': self.config["params_reg"]['params_gb']['learning_rate'],
        }

        params_reg_xgb = {
            'classifier__n_estimators': self.config["params_reg"]['params_xgb']['n_estimators'],
            'classifier__max_depth': self.config["params_reg"]['params_xgb']['max_depth'],
            'classifier__subsample': self.config["params_reg"]['params_xgb']['subsample'],
            'classifier__learning_rate': self.config["params_reg"]['params_xgb']['learning_rate'],
            'classifier__num_boost_rounds': self.config["params_reg"]['params_xgb']
                                                       ['num_boost_rounds'],
            'classifier__early_stopping_rounds': self.config["params_reg"]['params_xgb']
                                                            ['early_stopping_rounds'],
        }

        params_reg_lgbm = {
            'classifier__n_estimators': self.config["params_reg"]['params_lgbm']['n_estimators'],
            'classifier__max_depth': self.config["params_reg"]['params_lgbm']['max_depth'],
            'classifier__max_features': self.config["params_reg"]['params_lgbm']['max_features'],
            'classifier__min_samples_leaf': self.config["params_reg"]['params_lgbm']
                                                       ['min_samples_leaf'],
            'classifier__min_samples_split': self.config["params_reg"]['params_lgbm']
                                                        ['min_samples_split'],
        }

        # Initialize different scalers
        dict_scaler = {
            "Standard": StandardScaler(),
            "MinMax": MinMaxScaler(),
            "Normalizer": Normalizer(),
        }

        # On ajoute les scalers dans la liste des différents hyperparametres à ajuster pour algos
        # La liste des scalers est définie dans les parametres du modele
        for param in [params_reg_linear, params_reg_ridge, params_reg_lasso, params_reg_elastic,
                      params_reg_tree, params_reg_forest, params_reg_gb, params_reg_xgb,
                      params_reg_lgbm]:

            param['preprocessor__continuous__scaler'] = []
            for scaler in self.config["pipeline_preprocessing"]['scaler']:
                param['preprocessor__continuous__scaler'].append(dict_scaler[scaler])

        # on initialise le dictionnaire sur lequel on va réaliser la gridsearch
        self.dict_algo = {
                "Linear Regression": (LinearRegression(fit_intercept=True), params_reg_linear),
                "Ridge": (Ridge(fit_intercept=True), params_reg_ridge),
                "Lasso": (Lasso(fit_intercept=True,
                                precompute=True, warm_start=True), params_reg_lasso),
                "Elastic Net": (ElasticNet(fit_intercept=True,
                                precompute=True, warm_start=True), params_reg_elastic),
                "Decision Tree": (DecisionTreeRegressor(), params_reg_tree),
                "Random Forest": (RandomForestRegressor(), params_reg_forest),
                "Gradient Boosting": (GradientBoostingRegressor(warm_start=True), params_reg_gb),
                "Extreme Gradient Boosting": (XGBRegressor(warm_start=True), params_reg_xgb),
                # pour l'instant on a un probleme d'installation avec light gbm
                # "Light GBM": (LGBMRegressor(), params_reg_lgbm),
            }

        self.dict_algo = {key: self.dict_algo[key] for key in self.config["input_models"]['reg']}

        # on log le dictionnaire
        print(self.dict_algo)
        return None

    def init_metrics(self, selection_metric_classification='AUC',
                     selection_metric_regression='RMSE'):

        # on initialise la liste des métriques à partir desquelles réaliser l'étude

        list_classification_metrics = {"AUC": "roc_auc",
                                       "Accuracy": 'accuracy',
                                       'F1': 'f1',
                                       "Precision": 'precision',
                                       "Recall": 'recall'}

        list_regression_metrics = {"RMSE": "neg_root_mean_squared_error",
                                   "MAE": 'neg_mean_absolute_error',
                                   'MAPE': 'neg_mean_absolute_percentage_error',
                                   "Explained Variance": 'explained_variance',
                                   "R2 (not ajusted)": "r2",
                                   "Max Residual Error": 'max_error',
                                   }

        if self.is_regression is True:
            self.list_metrics = list_regression_metrics
            self.selection_metrics = selection_metric_regression
        else:
            self.list_metrics = list_classification_metrics
            self.selection_metrics = selection_metric_classification

        return

    def init_gridsearch_results(self):

        # On initialise les entrées du fichier résultat de la recherche des hyperparameters
        hyperparam_results = {
            "Algo Name": [],
            "Preprocessor Name": [],
            f"GridSearch Test: Mean {self.selection_metrics} Score (selection score)": [],
            f"GridSearch Test: Std {self.selection_metrics} Score": [],
            f"GridSearch Test: Mean {self.selection_metrics} Score (selection score) Fold": [],
            f"GridSearch Test: Std {self.selection_metrics} Score Fold": [],
            "GridSearch Mean Training Time (s)": [],
            "GridSearch Mean Training Time (s) Fold": [],
            "GridSearch Best Parameters": [],
            "GridSearch Best Algo": []
                              }

        # On initialise les entrées du fichier résultat
        detailed_results = {"Algo Name": [],
                            "Preprocessor Name": [],
                            "y_pred_best_hyper": [],
                            "y_true": [],
                            "X_true": []
                            }

        self.hyperparam_results = hyperparam_results
        self.detailed_results = detailed_results

        return

    def return_predictions(self):

        # on retourne les predictions
        y_true = self.detailed_results["y_true"].iloc[0]
        X_true = self.detailed_results["X_true"].iloc[0]

        self.models_predictions = pd.DataFrame(X_true)
        self.models_predictions['y_true'] = y_true

        for i, name in enumerate(self.detailed_results.index):
            self.models_predictions[f'y_pred_{name}'] = (
                self.detailed_results.loc[name, 'y_pred_best_hyper']
            )

        return

    def summary_error(self):

        self.detailed_results["Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.array(x["y_true"]) - np.array(x["y_pred_best_hyper"]))**2,
            axis=1)
        self.detailed_results["Root Mean Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: root_mean_squared_error(x["y_true"], x["y_pred_best_hyper"]),
            axis=1)
        self.detailed_results["Root Std Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.std(x["Squared Error (Test)"]))**0.5,
            axis=1)
        self.detailed_results["Root min Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.min(x["Squared Error (Test)"])**0.5),
            axis=1)
        self.detailed_results["Root Q1 Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.percentile(x["Squared Error (Test)"], 25)**0.5),
            axis=1)
        self.detailed_results["Root Median Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.percentile(x["Squared Error (Test)"], 50)**0.5),
            axis=1)
        self.detailed_results["Root Q3 Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.percentile(x["Squared Error (Test)"], 75)**0.5),
            axis=1)
        self.detailed_results["Root max Squared Error (Test)"] = self.detailed_results.apply(
            lambda x: (np.max(x["Squared Error (Test)"])**0.5),
            axis=1)

        self.detailed_results["Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: abs(np.array(x["y_true"]) - np.array(x["y_pred_best_hyper"])),
            axis=1)
        self.detailed_results["Mean Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: mean_absolute_error(x["y_true"], x["y_pred_best_hyper"]),
            axis=1)
        self.detailed_results["Std Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: np.std(x["Absolute Error (Test)"]),
            axis=1)
        self.detailed_results["min Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: np.min(x["Absolute Error (Test)"]),
            axis=1)
        self.detailed_results["Q1 Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x:  np.percentile(x["Absolute Error (Test)"], 25),
            axis=1)
        self.detailed_results["Median Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: np.percentile(x["Absolute Error (Test)"], 50),
            axis=1)
        self.detailed_results["Q3 Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: np.percentile(x["Absolute Error (Test)"], 75),
            axis=1)
        self.detailed_results["max Absolute Error (Test)"] = self.detailed_results.apply(
            lambda x: np.max(x["Absolute Error (Test)"]),
            axis=1)

        self.detailed_results["Absolute Percentage Error (Test)"] = (
            self.detailed_results.apply(
                lambda x: (abs(np.array(x["y_true"]) - np.array(x["y_pred_best_hyper"])))
                / np.array(x["y_true"])*100,
                axis=1)
            )
        self.detailed_results["Mean Absolute Percentage Error (Test)"] = (
            self.detailed_results.apply(lambda x: np.mean(x["Absolute Percentage Error (Test)"]),
                                        axis=1)
            )
        self.detailed_results["Std Absolute Percentage Error (Test)"] = self.detailed_results.apply(
            lambda x: np.std(x["Absolute Percentage Error (Test)"]),
            axis=1)
        self.detailed_results["min Absolute Percentage Error (Test)"] = self.detailed_results.apply(
            lambda x: np.min(x["Absolute Percentage Error (Test)"]),
            axis=1)
        self.detailed_results["Q1 Absolute Percentage Error (Test)"] = self.detailed_results.apply(
            lambda x:  np.percentile(x["Absolute Percentage Error (Test)"], 25),
            axis=1)
        self.detailed_results["Median Absolute Percentage Error (Test)"] = (
            self.detailed_results.apply(
                lambda x: np.percentile(x["Absolute Percentage Error (Test)"], 50),
                axis=1)
            )
        self.detailed_results["Q3 Absolute Percentage Error (Test)"] = self.detailed_results.apply(
            lambda x: np.percentile(x["Absolute Percentage Error (Test)"], 75),
            axis=1)
        self.detailed_results["max Absolute Percentage Error (Test)"] = self.detailed_results.apply(
            lambda x: np.max(x["Absolute Percentage Error (Test)"]),
            axis=1)

        self.detailed_results.set_index(['Algo Name', 'Preprocessor Name'],
                                        inplace=True, drop=False)

        return

    def plot_error_distrib(self, n_first=10):

        print_log('....Plotting Error Distribution...')

        fig_L2, ax2 = plt.subplots(layout="constrained")
        fig_L1, ax1 = plt.subplots(layout="constrained")
        fig_3, ax3 = plt.subplots(layout="constrained")
        fig_L2.suptitle('L2 Error Boxplot', fontsize=14, fontweight='bold')
        fig_L1.suptitle('L1 Error Boxplot', fontsize=14, fontweight='bold')
        fig_3.suptitle('Absolute Percentage Error Boxplot', fontsize=14, fontweight='bold')

        data_plot = self.detailed_results[
            ['Algo Name', 'Preprocessor Name', "Absolute Error (Test)",
             "Mean Absolute Error (Test)", "Squared Error (Test)",
             "Root Mean Squared Error (Test)", "Mean Absolute Percentage Error (Test)",
             "Absolute Percentage Error (Test)", "Root max Squared Error (Test)",
             "max Absolute Error (Test)", "max Absolute Error (Test)"
             ]]

        data_L2 = data_plot.sort_values(by=["Root Mean Squared Error (Test)"], ascending=False)
        data_L1 = data_plot.sort_values(by=["Mean Absolute Error (Test)"], ascending=False)
        data_3 = data_plot.sort_values(by=["Mean Absolute Percentage Error (Test)"],
                                       ascending=False)

        ax2.boxplot(
                data_L2["Squared Error (Test)"],
                labels=data_L2['Algo Name'] + '_' + data_L2['Preprocessor Name'],
                vert=False
                )
        ax2.set_xlim([0, data_L2["Squared Error (Test)"].apply(max).sort_values()[:2].max()])

        ax1.boxplot(
                data_L1["Absolute Error (Test)"],
                labels=data_L1['Algo Name'] + '_' + data_L1['Preprocessor Name'],
                vert=False
                )
        print(data_L1)
        # ax1.set_xlim([0, data_L1.sort_values(by=["Mean Absolute Error (Test)"])
        # ["max Absolute Error (Test)"].iloc[0]])
        ax1.set_xlim([0, data_L1["Absolute Error (Test)"].apply(max).sort_values()[:2].max()])

        ax3.boxplot(
                data_3["Absolute Percentage Error (Test)"],
                labels=data_3['Algo Name'] + '_' + data_3['Preprocessor Name'],
                vert=False
                )

        ax3.set_xlim(
            [0, data_3["Absolute Percentage Error (Test)"].apply(max).sort_values()[:2].max()]
            )

        fig_L2.savefig("error/L2_error_boxplot.png")
        fig_L1.savefig("error/L1_error_boxplot.png")
        fig_3.savefig("error/APE_error_boxplot.png")
        plt.close()

        return

    def plot_residuals(self, n_first=10):

        print_log('....Plotting Residuals...')

        # Study Residuals
        for i, algo in enumerate(self.detailed_results['Algo Name']):
            for j, preproc in enumerate(self.detailed_results['Preprocessor Name']):
                if i < n_first:

                    y_true = self.detailed_results.loc[
                        (self.detailed_results["Algo Name"] == algo) &
                        (self.detailed_results["Preprocessor Name"] == preproc),
                        "y_true"].values[0]
                    y_pred = self.detailed_results.loc[
                        (self.detailed_results["Algo Name"] == algo) &
                        (self.detailed_results["Preprocessor Name"] == preproc),
                        "y_pred_best_hyper"].values[0]

                    fig, axs = plt.subplots(ncols=2)

                    PredictionErrorDisplay.from_predictions(
                        y_true=np.array(y_true),
                        y_pred=np.array(y_pred),
                        kind="actual_vs_predicted",
                        ax=axs[0],
                        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
                        line_kwargs={"color": "tab:red"},
                        )
                    axs[0].set_title("Actual vs. Predicted values")

                    PredictionErrorDisplay.from_predictions(
                        y_true=np.array(y_true),
                        y_pred=np.array(y_pred),
                        kind="residual_vs_predicted",
                        ax=axs[1],
                        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
                        line_kwargs={"color": "tab:red"},
                        )
                    axs[1].set_title("Residuals vs. Predicted Values")
                    fig.suptitle(f"Plotting cross-validated predictions - {algo} {preproc}")

                    fig.savefig(f"residuals/{algo}_{preproc}.png")
                    plt.close()

        return

    def compare(self):

        self.init_metrics()
        self.init_gridsearch_results()

        # on sort la boucle kfold pour s'assurer que les algo sont bien
        # entraintés sur les meme folds
        kf = KFold(n_splits=self.config["settings"]['cv_test'], shuffle=True, random_state=42)

        for name, (classifier, params) in self.dict_algo.items():
            print_log(f"\n {name} \n")

            for preproc_name, preproc in self._preprocesseurs.items():
                print_log(f"\n {preproc_name} \n")

                mean_grid_score = []
                std_grid_score = []
                mean_grid_time = []
                best_grid_param = []
                best_grid_algo = []
                y_pred_best_hyper = []
                y_true = []
                X_true = pd.DataFrame(columns=self.X.columns)

                # pour chaque fold on réalise un grid search puis on évalue le meilleur modele
                for train_index, test_index in kf.split(self.X, self.Y):
                    X_train_val, X_test, = self.X.loc[train_index], self.X.loc[test_index]
                    y_train_val, y_test = self.Y.loc[train_index], self.Y.loc[test_index]

                    pip = Pipeline(steps=[
                        ('preprocessor', preproc),
                        ('classifier', classifier)
                    ])

                    if params is not None:
                        grid = GridSearchCV(estimator=pip,
                                            param_grid=params,
                                            cv=self.config["grid_search"]["cv"],
                                            scoring=self.list_metrics,
                                            refit=self.selection_metrics,
                                            # verbose=1,
                                            n_jobs=-1,
                                            error_score='raise'
                                            ).fit(X_train_val, y_train_val)

                    # on sauve les résultats de la recherche des meilleurs hyperparameters
                    # on a pour chaque entrée une liste de K_fold valeurs.
                    mean_grid_score.append(round(grid.best_score_, 3))
                    std_grid_score.append(
                        round(grid.cv_results_[f'std_test_{self.selection_metrics}']
                                              [grid.best_index_], 3))
                    mean_grid_time.append(round(grid.refit_time_, 3))
                    best_grid_param.append(grid.best_params_)
                    best_grid_algo.append(grid.best_estimator_)

                    # residuals
                    y_pred_best_hyper.append(grid.best_estimator_.predict(X_test))
                    # y_pred_best_hyper.append(grid.best_estimator_.predict_proba(X_test)[:, 1])
                    # pour categoriel
                    y_true.append(y_test)
                    X_true = pd.concat([X_true, X_test])

                self.hyperparam_results["Algo Name"].append(f"{name}")
                self.hyperparam_results["Preprocessor Name"].append(f"{preproc_name}")
                self.hyperparam_results[
                    f"GridSearch Test: Mean {self.selection_metrics} Score "
                    "(selection score) Fold"].append(np.negative(mean_grid_score))
                self.hyperparam_results[
                    f"GridSearch Test: Std {self.selection_metrics} "
                    "Score Fold"].append(std_grid_score)
                self.hyperparam_results[
                    "GridSearch Mean Training Time (s) Fold"].append(mean_grid_time)
                self.hyperparam_results["GridSearch Best Parameters"].append(best_grid_param)
                self.hyperparam_results["GridSearch Best Algo"].append(best_grid_algo)

                # on estime sur le meilleur modele sur le test
                # on rappelle que lorsque refit est entré comme parametre alors
                # grid_gb_fitted.score(X_test, y_test) =
                # roc_auc_score(y_test,best_gb.predict_proba(X_test)[:,1])

                self.detailed_results["Algo Name"].append(f"{name}")
                self.detailed_results["Preprocessor Name"].append(f"{preproc_name}")

                self.detailed_results["y_pred_best_hyper"].append(y_pred_best_hyper)
                self.detailed_results["y_true"].append(y_true)
                self.detailed_results["X_true"].append(X_true)

        # ON AGGREGE LES RESULTATS DE LA CROSS VALIDATION

        # on a reconstruit, pour toutes les données, 2 vecteurs: y true et y_pred avec le meilleur
        # set de hyperparameter
        # possible sur les K-1 fold de l'entraintement.
        # 1) pour chaque métrique, on estime min, max, Q1, Q2, Q3
        # 2) on étudie les résidus
        # on evalue les métriques qui nous intéressent
        self.hyperparam_results[
            f"GridSearch Test: Mean {self.selection_metrics}"
            " Score (selection score)"] = (
                np.mean(self.hyperparam_results[f"GridSearch Test: Mean {self.selection_metrics}"
                                                " Score (selection score) Fold"], axis=1)
                                        )
        self.hyperparam_results[f"GridSearch Test: Std {self.selection_metrics}"
                                " Score"] = (
                np.mean(self.hyperparam_results[f"GridSearch Test: Std {self.selection_metrics}"
                                                " Score Fold"], axis=1)
                                            )
        self.hyperparam_results["GridSearch Mean Training Time (s)"] = (
            np.mean(self.hyperparam_results["GridSearch Mean Training Time (s) Fold"], axis=1)
            )
        self.hyperparam_results = pd.DataFrame(self.hyperparam_results)
        self.hyperparam_results.sort_values(
            by=[f"GridSearch Test: Mean {self.selection_metrics} Score (selection score)"],
            ascending=True, inplace=True)
        self.hyperparam_results.set_index(['Algo Name', 'Preprocessor Name'], inplace=True)

        self.detailed_results = pd.DataFrame(self.detailed_results)
        self.detailed_results["y_pred_best_hyper"] = (
            self.detailed_results["y_pred_best_hyper"].apply(lambda x: flatten(x))
            )
        self.detailed_results["y_true"] = (
            self.detailed_results["y_true"].apply(lambda x: flatten(x))
            )

        self.summary_error()
        self.detailed_results.sort_values(
            by="Root Mean Squared Error (Test)", ascending=True, inplace=True)

        print_log("\n ---------- HYPERPARAMETER RESULTS ------------- \n")
        pd.set_option('display.max_columns', None)  # Display the wholde dataframe
        print_log(self.hyperparam_results.drop(
            ["GridSearch Best Algo",
             f"GridSearch Test: Mean {self.selection_metrics} Score (selection score) Fold",
             f"GridSearch Test: Std {self.selection_metrics} Score Fold",
             "GridSearch Mean Training Time (s) Fold"], axis=1).round(3))

        print_log('\n ---------- DETAILED CROSS VALIDATION RESULTS ------- \n')
        pd.set_option('display.max_columns', None)  # Display the wholde dataframe
        print_log(self.detailed_results.drop(
            columns=["Algo Name", "Preprocessor Name", "y_pred_best_hyper",
                     "y_true", "X_true", "Squared Error (Test)", "Absolute Error (Test)",
                     "Absolute Percentage Error (Test)"], axis=1
                     ).round(3))

        # Ici on a selectionné par Grid Search les meilleurs parametre pour les algos étudiés.
        # on selectionne ensuite le meilleur algo définit par CV
        best_algo_index = self.detailed_results["Root Mean Squared Error (Test)"].idxmin()
        best_results = self.detailed_results.loc[best_algo_index]
        self.best_algo = best_results['Algo Name']
        self.best_preproc = best_results["Preprocessor Name"]

        print_log("\n ------- SUMMARY -------- \n ")
        print_log(
            f"The best model is {best_algo_index}"
            )

        self.return_predictions()

        self.plot_error_distrib()

        # if self.study_residuals is True:
        self.plot_residuals()
        self.return_errorfiles()

        # if save algo
        self.train_and_save_best_algo()

        return

    def return_errorfiles(self):

        self.detailed_results.drop(
            columns=["Algo Name", "Preprocessor Name", "y_pred_best_hyper",
                     "y_true", "X_true", "Squared Error (Test)", "Absolute Error (Test)",
                     "Absolute Percentage Error (Test)"], axis=1
                     ).round(3).to_csv("results/detailed_results.csv")
        self.hyperparam_results.to_csv("results/hyperparam_results.csv")
        self.models_predictions.to_csv("results/models_predictions.csv", index=False)

        return

    def plot_linear_coef_variability(self, n_first=10):

        # TO BE DONE
        return

    def train_and_save_best_algo(self):

        # on cherche sur le jeu de données optimal les meilleurs hyperparametres
        # sur je jeu de données total, on sauvegarde le modele

        best_pip = Pipeline(steps=[
                        ('preprocessor', self._preprocesseurs[self.best_preproc]),
                        ('classifier', self.dict_algo[self.best_algo][0])
                    ])

        best_grid = GridSearchCV(estimator=best_pip,
                                 param_grid=self.dict_algo[self.best_algo][1],
                                 cv=self.config["grid_search"]["cv"],
                                 scoring=self.list_metrics,
                                 refit=self.selection_metrics,
                                 # verbose=1,
                                 n_jobs=-1,
                                 error_score='raise'
                                 ).fit(self.X, self.Y)

        pd.DataFrame(best_grid.best_params_,
                     index=['Best Hyperparameters']
                     ).to_csv('results/best_model_best_hyperparam.csv')

        # to do LEARN BEST ALGO
        with open(
            f'best_algo/best_algo_{self.best_algo}_{self.best_preproc}_pickle.obj', 'wb'
                 ) as f:
            pickle.dump(best_grid.best_estimator_, f)


# on lance la fonction depuis le terminal
if __name__ == "__main__":

    """
    options
        -classif or -regression
        -save

    argument
        chemin du nom de fichier du dataset NETTOYE:
           - on traite les valeurs manquantes soit comme des manquantes (na),
           - on supprime les champs avec troip de valeurs manquantes
           - on met les bon types
    """

    assert sys.argv[2] in ["-regression", "-classification"], (
        "l'option est mauvaise. Veuillez séléctionner '-regression' ou '-classification' "
        )

    # assert sys.argv[2] in [None,"-save"], "l'option est mauvaise.
    # Veuillez séléctionner 'None' ou '-save' "

    # juste pour débugger facilement - les dataset CLEAN
    # data = "data/dfbase.csv"
    # data_cat = "data/spam_clean.csv"
    # data_cat_test = "data/data_test_clean.csv"

    data = sys.argv[1]

    df_clean = pd.read_csv('data/' + data)

    study_residuals = True if sys.argv[3] == "-residuals" else False

    # on nettoie ce qu'il se trouve dans le fichier results
    open('results/results.txt', 'w').close()

    if sys.argv[2] == "-regression":
        model_comparator = ModelComparator(df_clean, is_regression=True,
                                           study_residuals=study_residuals)
        results = model_comparator.compare()

    elif sys.argv[2] == "-classification":
        df_clean = pd.read_csv(data_cat)
        print_log(df_clean.head())

        model_comparator = ModelComparator(df_clean, is_regression=False,
                                           study_residuals=study_residuals)
        results = model_comparator.compare()

    # if sys.argv[2] == "-save":
    #    model_comparator.save()
