# utils.py
import pandas as pd
import numpy as np

def load_data(filename: str, exclude_cols=None, category_cols=None, price_cols=None, bool_cols=None):
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
    df = clean_df(df, price_cols=price_cols, bool_cols=bool_cols)

    # eliminate outliers, log transform price
    df = df[(df['price'] > 0) & (df['price'] < df['price'].quantile(0.99))]
    df['price'] = np.log1p(df['price'])

    if category_cols:
        df = encode_categorical(df, category_cols=category_cols)

    df.reset_index(drop=True, inplace=True)

    return df


def clean_df(df, price_cols=None, bool_cols=None):
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

    # handle price columns
    if price_cols:
        for col in price_cols:
            df[col] = (
                df[col]
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

    if bool_cols:
        df = str_to_bool(df, bool_cols)

    return df

def str_to_bool(df, bool_cols):
    """ 
    Takes a mapping of {col: list} where index 0 is the 
    label corresponding with binary value 0.
    """
    for col, lst in bool_cols.items():
        df[col] = df[col].astype(str).str.strip().map({
            lst[0]: 0,
            lst[1]: 1
        })

    return df

def encode_categorical(df, category_cols: list):
    """
    Handle our categorical columns, encode them
    numerically to be understood by the model
    """
    for col in category_cols:
        vals = list(df[col].str.strip().unique())
        nums = [i for i, val in enumerate(vals)]
        mapping = dict(zip(vals, nums))
        df[col] = df[col].str.strip().map(mapping)
        df[col] = df[col].astype(int)

    return df

def one_hot_encode(df, category_cols: list):
    for col in category_cols:
        print("CATEGORY COL: ", col)
        print(df[col].unique())
    df = pd.get_dummies(df, columns=category_cols, drop_first=True)
    return df