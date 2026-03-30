# models.py
import pandas as pd


class AirBNBModel:
    def __init__(self, model):
        self.model = model

    def clean_df(self, df):
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
        df['price'] = df['price'].str.strip().str.replace('$', '', regex=False
                                            ).str.replace(',', '', regex=False
                                            ).astype(float)
        df['service_fee'] = df['service_fee'].str.strip().str.replace('$', '', regex=False
                                                        ).str.replace(',', '', regex=False
                                                        ).astype(float)
            
        return df

    def load_data(self, filename: str, exclude_cols=None):
        """
        Load data from file in directory. Optional
        parameter to exclude columns from your data 
        (particularly built for handling text data).
        
        filename: str, file directory path
        exclude_cols: list, defaulted to none, takes
        list of strings of columns to not include in
        final output.
        """
        airbnb_df = pd.read_csv(filename, low_memory=False)
        if exclude_cols:
            airbnb_df.drop(columns=exclude_cols, axis=1, inplace=True)

        # drop na values
        airbnb_df.dropna(inplace=True)

        airbnb_df = self.clean_df(airbnb_df)

        return airbnb_df
