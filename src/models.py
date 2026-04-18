# models.py

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
import numpy as np
import utils


class AirBNBModel(object):
    """
    Template class for all models. 
    """
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.scaler = StandardScaler()

    def scale_data(self, X_train, X_test, y_train, y_test):
        """ Standard scale X, log transform y """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled

    def run_regression(self, df, y_col: str, test_size=0.2):
        """
        Fit, train & evaluate regression models.
        """
        df_X = df.drop(columns=y_col)
        df_y = df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=self.random_state)

        # scale features and target to standard scale
        X_train, X_test = self.scale_data(X_train, X_test, y_train, y_test)
        
        candidates = {
            "Linear Regression" : LinearRegression(),
        }

        fitted = {}
        print("\nREGRESSION (target: price)")

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            print(f"\n{name}")
            print(f"MAE (mean absolute error): {mae:.2f}")
            print(f"RMSE (root mean squre error): {rmse:.2f}")
            print(f"R^2: {r2:.4f}")
            fitted[name] = (model, preds)

        return fitted
    
    def run_classification(self, df, y_col: str, test_size=0.2, **kwargs):
        """
        Fit, train & evaluate classification models.
        """
        df_X = df.drop(columns=y_col)
        df_y = df[y_col]

        # split y_col into classes
        df_y, class_count = utils.data_to_classes(
            df_y,
            bins=[0, 100, 200, 400, np.inf],
            labels=['budget', 'mid', 'upper', 'luxury'], 
            log_transform=True
        )

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=self.random_state)

        # scale features and target to standard scale
        X_train, X_test = self.scale_data(X_train, X_test, y_train, y_test)

        fitted = {}

        print("\nCLASSIFICATION (target: price)")

        # mlp hidden state dims
        n_hid_neurons = kwargs.get("n_hid_neurons", 100)
        n_hid_layers = kwargs.get("n_hid_layers", 2)
        hidden_state = tuple([n_hid_neurons] * n_hid_layers)

        candidates = {
            "Logistic Regression": LogisticRegression(),
            "Multilayer Perceptron": MLPClassifier(
                hidden_layer_sizes=hidden_state,
                activation="relu",
                solver="sgd"
            ),
        }

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
            f1_wt = f1_score(y_test, preds, average="weighted", zero_division=0)
            print(f"\n{name}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
            print(f"F1 (wtd): {f1_wt:.4f}")
            print(classification_report(y_test, preds, zero_division=0))
            fitted[name] = (model, preds)

        return fitted
    
    def run_ensemble(self, df, y_col: str, test_size=0.2, **kwargs):
        """
        Fit, train & evaluate ensemble models.
        """
        df_X = df.drop(columns=y_col)
        df_y = df[y_col]

        # split y_col into classes
        df_y, class_count = utils.data_to_classes(
            df_y,
            bins=[0, 100, 200, 400, np.inf],
            labels=['budget', 'mid', 'upper', 'luxury'], 
            log_transform=True
        )

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=self.random_state)

        # scale features and target to standard scale
        X_train, X_test = self.scale_data(X_train, X_test, y_train, y_test)

        fitted = {}

        print("\nENSEMBLE (target: price)")

        # random forest dims
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", 5)

        candidates = {
            "Random Forest": RandomForestClassifier(
                n_estimators=n_estimators, 
                criterion="gini",
                max_depth=max_depth
            )
        }

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
            f1_wt = f1_score(y_test, preds, average="weighted", zero_division=0)
            print(f"\n{name}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
            print(f"F1 (wtd): {f1_wt:.4f}")
            print(classification_report(y_test, preds, zero_division=0))
            fitted[name] = (model, preds)