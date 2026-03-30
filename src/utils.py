# utils.py
import pandas as pd

def load_data(filename: str, exclude_cols=None, category_cols=None, price_cols=None):
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
    df = clean_df(df, price_cols=price_cols)
    print(df.dtypes)

    if category_cols:
        df = encode_categorical(df, category_cols=category_cols)

    print(df.dtypes.value_counts())

    return df

def clean_df(df, price_cols=None):
    """
    Cleaning df header names to follow same 
    conventions, handling mixed dtypes in cols.
    """

    update_mapping = {}
    for col in df.columns:
        update_mapping.update({
            col: col.lower().replace(" ", "_")
        })
        if col not in price_cols and df[col].dtype == object:
            df[col] = df[col].astype(str)

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

    return df

def encode_categorical(df, category_cols: list):
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
            df[col] = df[col].astype(int)
            print("CATEGORY COL: ", col)
            print(df[col].unique())

        else:
            print(f"Column {col} dtype incompatible; is it really categorical?")

    return df

def one_hot_encode(df, category_cols: list):
    for col in category_cols:
        print("CATEGORY COL: ", col)
        print(df[col].unique())
    df = pd.get_dummies(df, columns=category_cols, drop_first=True)
    return df