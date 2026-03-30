
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("airbnb.csv")

# Clean price column
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df = df.dropna(subset=['price', 'room type'])

# Get top 10 most frequent neighborhoods
top_neighborhoods = df['neighbourhood'].value_counts().head(10).index

df_top = df[df['neighbourhood'].isin(top_neighborhoods)]

# Compute average price
avg_price_neigh = df_top.groupby('neighbourhood')['price'].mean().reset_index()

# Sort for cleaner visualization
avg_price_neigh = avg_price_neigh.sort_values(by='price', ascending=False)

# price by neighborhood
plt.figure(figsize=(10,6))
sns.barplot(x='price', y='neighbourhood', data=avg_price_neigh)
plt.title("Average Price by Neighborhood (Top 10)")
plt.xlabel("Average Price")
plt.ylabel("Neighborhood")
plt.show()

# price by room type
plt.figure()
sns.boxplot(x='room type', y='price', data=df)
plt.title("Price Distribution by Room Type")
plt.xlabel("Room Type")
plt.ylabel("Price")
plt.xticks(rotation=30)
plt.show()