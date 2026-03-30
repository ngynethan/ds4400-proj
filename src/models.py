# models.py
import pandas as pd
from sklearn.linear_model import LinearRegression

class AirBNBModel(object):
    """
    Template class for all models. 
    """
    def __init__(self):
        self.model = None

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
        airbnb_df = pd.read_csv(filename, low_memory=False)
        if exclude_cols:
            airbnb_df.drop(columns=exclude_cols, axis=1, inplace=True)

        # drop na values
        airbnb_df.dropna(inplace=True)
        airbnb_df.reset_index(drop=True, inplace=True)
        airbnb_df = self.clean_df(airbnb_df, price_cols=price_cols)

        if category_cols:
            airbnb_df = self.encode_categorical(airbnb_df, category_cols=category_cols)

        return airbnb_df
    
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
    
    def train_model(self):
        pass
    
    def evaluate(self, model):
        pass

class LinearRegressionModel(AirBNBModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
