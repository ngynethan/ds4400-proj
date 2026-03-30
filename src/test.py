# test.py
from models import AirBNBModel
from sklearn.linear_model import LinearRegression

def main():
    model = AirBNBModel(LinearRegression)

    # ditch irrelevant textual data and locations
    exclude_cols = [
        "id", "NAME", "host id", "host name", 
        "neighbourhood group", "neighbourhood", 
        "lat", "long", "country", "country code", 
        "house_rules", "license"
    ]
    df = model.load_data("data/airbnb.csv", exclude_cols=exclude_cols)
    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    main()