# models.py
import pandas as pd
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
    def __init__(self):
        self.model = None
        self.random_state = random_state

    def load_data(self, filename: str, exclude_cols=None, category_cols=None, price_cols=None):
        """
        Load data from file in directory. Optional parameter to exclude 
        columns from your data (particularly built for handling text data).
        
        filename: str, file directory path
        exclude_cols: list, defaulted to None, takes list of strings of columns 
        to not include in final output.
        categorical_cols: list, defaults to None, takes list of strings of 
        columns to numerically encode.
        """
        df = pd.read_csv(filename, low_memory=False)
        if exclude_cols:
            df.drop(columns=exclude_cols, axis=1, inplace=True)

        # drop na values
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = self.clean_df(df, price_cols=price_cols)

        if category_cols:
            df = self.encode_categorical(df, category_cols=category_cols)

        return df
    
    def clean_df(self, df, price_cols=None):
        """
        Cleaning df header names to follow same 
        conventions, handling mixed dtypes in cols.
        """
        update_mapping = {}
        for col in df.columns:
            update_mapping.update({
                col: col.lower().replace(" ", "_")
            })

        df.rename(columns=update_mapping, inplace=True)

        # handle price tags
        if price_cols:
            for col in price_cols:
                df[col] = df[col].str.strip().str.replace('$', '', regex=False
                                ).str.replace(',', '', regex=False).astype(float)
        return df
    
    def encode_categorical(self, df, category_cols: list):
        """
        Handle our categorical columns, encode them
        numerically to be understood by the model
        """
        for col in category_cols:
            if type(df[col].iloc[0]) is bool:
                df[col] = df[col].astype(int)
            elif type(df[col].iloc[0]) is str:
                vals = list(df[col].str.strip().unique())
                nums = [i for i, val in enumerate(vals)]
                mapping = dict(zip(vals, nums))
                df[col] = df[col].str.strip().map(mapping)

            else:
                print(f"Column {col} dtype incompatible; is it really categorical?")

        return df
    
    def run_regression(self, df, y_col: str, test_size=0.2):
        """
        Fit, train & evaluate regression models.
        """
        df_X = df.drop(columns=y_col)
        df_y = df[y_col]

        X_train, y_train, X_test, y_test = train_test_split(df_X, df_y, test_size=test_size, random_state=self.random_state)

        candidates = {
            "Linear Regression" : LinearRegression(),
        }

        fitted = {}
        print("REGRESSION (target: total minutes)")

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            r2 = r2_score(y_test, preds)
            print(f"\n{name}")
            print(f"MAE (mean absolute error): {mae:.1f} min")
            print(f"RMSE (root mean squre error): {rmse:.1f} min")
            print(f"R^2: {r2:.4f}")
            fitted[name] = (model, preds)

        return fitted

