# models.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay,
)

class AirBNBModel(object):
    """
    Template class for all models. 
    """
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state

    def run_regression(self, df, y_col: str, test_size=0.2):
        """
        Fit, train & evaluate regression models.
        """
        df_X = df.drop(columns=y_col)
        df_y = df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=self.random_state)

        candidates = {
            "Linear Regression" : LinearRegression(),
        }

        fitted = {}
        print("REGRESSION (target: price)")

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

