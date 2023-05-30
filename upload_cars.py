import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['mydb']
collection = db['cars']

# Read the CSV file into a DataFrame
df = pd.read_csv('C:/Users/kaczm/OneDrive/Dokumenty/vehicles.csv')

# Filter rows with non-null values in the required columns
df = df[['manufacturer', 'model', 'year', 'paint_color', 'price']].dropna()

# Rename the columns
df.rename(columns={'manufacturer': 'brand', 'paint_color': 'color'}, inplace=True)

# Convert DataFrame to a list of dictionaries
cars = df.to_dict(orient='records')

# Insert cars into the database
collection.insert_many(cars)