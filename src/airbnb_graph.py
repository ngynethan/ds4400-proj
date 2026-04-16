
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_by_neighborhood(data):
    plt.figure(figsize=(10,6), dpi=200)
    sns.barplot(x='price', y='neighbourhood', data=data)
    plt.title("Average Price by Neighborhood (Top 10)")
    plt.xlabel("Average Price")
    plt.ylabel("Neighborhood")
    plt.tight_layout()
    plt.savefig("viz/price_by_neigh.png")

def plot_price_by_room(df):
    plt.figure()
    sns.boxplot(x='room_type', y='price', data=df)
    plt.title("Price Distribution by Room Type")
    plt.xlabel("Room Type")
    plt.ylabel("Price")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("viz/price_by_room.png")

def plot_price_distribution(df):
    plt.figure(figsize=(10,6), dpi=200)
    plt.hist(df["price"].clip(upper=2500), bins=60)
    plt.title("NYC Airbnb Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("viz/price_dist.png")


def main():
    df = pd.read_csv("data/AB_NYC_2019.csv")

    # Clean price column
    # df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    df = df.dropna(subset=['price', 'room_type'])

    # Get top 10 most frequent neighborhoods
    top_neighborhoods = df['neighbourhood'].value_counts().head(10).index

    df_top = df[df['neighbourhood'].isin(top_neighborhoods)]

    # Compute average price
    avg_price_neigh = df_top.groupby('neighbourhood')['price'].mean().reset_index()

    # Sort for cleaner visualization
    avg_price_neigh = avg_price_neigh.sort_values(by='price', ascending=False)

    # plot visualizations
    plot_price_by_neighborhood(avg_price_neigh)
    plot_price_by_room(df)
    # plot_price_distribution(df)

if __name__ == "__main__":
    main()