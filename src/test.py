# test.py

from models import AirBNBModel
from utils import load_data
import numpy as np
from airbnb_graph import plot_price_distribution

def main():
    model = AirBNBModel()

    # ditch irrelevant textual data
    exclude_cols = [
        "id", "name", "host_id", "host_name", "last_review"
    ]
    # define category cols
    category_cols = [
        "room_type", "neighbourhood",
        "neighbourhood_group", 
    ]

    airbnb_df = load_data(
        "data/AB_NYC_2019.csv",
        exclude_cols=exclude_cols, 
        category_cols=category_cols, 
    )
    plot_price_distribution(airbnb_df)

    airbnb_df.to_csv("data/CLEAN_DF.csv")
    
    model.run_regression(airbnb_df, "price")

    # run
    model.run_classification(airbnb_df, "price")

if __name__ == "__main__":
    main()