# test.py
from models import LinearRegressionModel

def main():
    model = LinearRegressionModel()

    # ditch irrelevant textual data and locations
    exclude_cols = [
        "id", "NAME", "host id", "host name", 
        "neighbourhood group", "neighbourhood", 
        "lat", "long", "country", "country code", 
        "house_rules", "license", "last review"
    ]
    # define category cols
    category_cols = [
        "host_identity_verified", "instant_bookable", 
        "cancellation_policy", "room_type",
    ]
    # price cols
    price_cols = ["price", "service_fee"]

    df = model.load_data(
        "data/airbnb.csv", exclude_cols=exclude_cols, 
        category_cols=category_cols, price_cols=price_cols
    )
    print(df.head())
    print(df.shape)
    print(df["instant_bookable"])

if __name__ == "__main__":
    main()