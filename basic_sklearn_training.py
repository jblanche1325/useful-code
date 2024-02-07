import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score

from catboost import CatBoostClassifier
import lightgbm as lgb

#from src.utils import save_object, evaluate_models

#rom sklearn.metrics import recall_score
#rom sklearn.model_selection import GridSearchCV
#
#ef evaluate_models(X_train, y_train, X_test, y_test, models, params):
#   report = {}
#
#   for i in range(len(list(models))):
#       model = list(models.values())[i]
#       para = params[list(models.keys())[i]]
#
#       gs = GridSearchCV(model, para, cv=3)
#       gs.fit(X_train, y_train)
#
#       model.set_params(**gs.best_params_)
#       model.fit(X_train, y_train)
#
#       y_train_pred = model.predict(X_train)
#       y_test_pred = model.predict(X_test)
#
#       train_model_score = recall_score(y_train, y_train_pred)
#       test_model_score = recall_score(y_test, y_test_pred)
#
#       report[list(models.keys())[i]] = test_model_score
#
#   return report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('<PATH>', 'model.pkl')  # Path to save model object

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):

        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models = {
            'Logistic Regression': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'SVM': SVC(),
            'Cat Boosting': CatBoostClassifier(silent=True),
            'Light GBM': lgb.LGBMClassifier()
        }

        params = {
            'Logistic Regression': {
                'penalty': [None, 'l2']#,
                #'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
            },
            'KNN': {
                'n_neighbors': [2, 3, 4, 5]
            },
            'Decision Tree': {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 1, 2],
                'max_features': [None, 'sqrt']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 250, 500],
                'max_features': [None, 'sqrt']
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]
            },
            'SVM': {
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2, 3]
            },
            'Cat Boosting': {
                'iterations': [100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]
            },
            'Light GBM': {
                'num_iterations': [100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
            }
        }

        model_report:dict = evaluate_models(X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            models=models,
                                            params=params
                                            )
        
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        #save_object(
        #    file_path=self.model_trainer_config.trained_model_file_path,
        #    obj=best_model
        #)

        predicted = best_model.predict(X_test)
        predicted_probabilities = best_model.predict_proba(X_test)
        recall = recall_score(y_test, predicted)

        return recall, best_model_name
