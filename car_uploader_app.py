from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

class CarUploader:
    def __init__(self, database_name='mydb', collection_name='cars', host='localhost', port=27017):
        self.database_name = database_name
        self.collection_name = collection_name
        self.host = host
        self.port = port

    def run(self):
        # Connect to MongoDB
        client = MongoClient(self.host, self.port)
        db = client[self.database_name]
        collection = db[self.collection_name]

        cars = []

        start = input("Do you want to input a car? (y/n): ")

        if start.lower() == 'y':
                
            while True:
                brand = input("Enter the brand of the car: ")
                model = input("Enter the model of the car: ")
                year = int(input("Enter the year of the car: "))
                color = input("Enter the color of the car: ")
                price = float(input("Enter the price of the car: "))

                car = {
                    'brand': brand,
                    'model': model,
                    'year': year,
                    'color': color,
                    'price': price
                }

                cars.append(car)


                choice = input("Do you want to input another car? (y/n): ")

                if choice.lower() != 'y':
                    break

                # Insert cars into the database
            collection.insert_many(cars)
            print("Cars uploaded successfully!")

        # Additional actions
        while True:
            action = input("Choose an action to perform:\n"
                           "1. Get average price of the car per brand\n"
                           "2. Get a histogram of car production year per brand\n"
                           "3. Get the most common color\n"
                           "4. Create a random forest regression model for price prediction\n"
                           "5. Exit\n"
                           "Enter the action number: ")

            if action == '1':
                self.get_average_price_per_brand(collection)
            elif action == '2':
                self.get_histogram_of_production_year_per_brand(collection)
            elif action == '3':
                self.get_most_common_color(collection)
            elif action == '4':

                brand_choice = input("Which brand do you wish to create the model for?: ")

                # Send notification to keep waiting
                print("Please wait. The model is working...")

                self.create_regression_model(collection, brand_choice)

                # Model processing completed
                print("Model processing completed!")

                # Wait for 5 seconds
                time.sleep(5)
            elif action == '5':
                break
            else:
                print("Invalid action choice.")
        print("Program exited. Goodbye!")


    def create_regression_model(self, collection, chosen_brand):

        data = list(collection.find({"brand": chosen_brand.lower()}, {'brand': 1, 'model': 1, 'year': 1, 'color': 1, 'price': 1}))
        df = pd.DataFrame(data)

        # Fill missing color values with 'Unknown'
        df['color'].fillna('Unknown', inplace=True)

        # Remove entries with unknown price
        df.dropna(subset=['price'], inplace=True)

        # Convert string values to lowercase
        df[['brand', 'model', 'color']] = df[['brand', 'model', 'color']].apply(lambda x: x.str.lower())

        X = df[['brand', 'model', 'year', 'color']]
        y = df['price']

        # Convert categorical variables to dummy/indicator variables
        X = pd.get_dummies(X)

        # Split the data into training and testing sets (80:20 ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Generate predictions for testing data
        y_pred = model.predict(X_test)

        # Plot regression fit for testing data
        plt.plot(y_test, y_pred, 'bo', alpha=0.5, label='Predicted Price')  # Use blue dots with opacity for predicted prices
        plt.plot(y_test, y_test, 'ro', label='Actual Price')  # Use red dots for actual prices
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Random Forest Regression Fit (Testing Data)')
        plt.legend()
        plt.show()

    def get_average_price_per_brand(self, collection):
        pipeline = [
            {"$group": {"_id": "$brand", "avg_price": {"$avg": "$price"}}}
        ]
        result = collection.aggregate(pipeline)

        print("Average price per brand:")
        for doc in result:
            print(f"Brand: {doc['_id']}, Average Price: {doc['avg_price']}")


    def get_histogram_of_production_year_per_brand(self, collection):
        pipeline = [
        {"$group": {"_id": {"brand": "$brand", "year": "$year"}, "count": {"$sum": 1}}}
    ]
        result = collection.aggregate(pipeline)

        brand_years = {}
        for doc in result:
            brand = doc['_id']['brand']
            year = doc['_id']['year']
            count = doc['count']

            if brand not in brand_years:
                brand_years[brand] = []

            brand_years[brand].extend([year] * count)

        # Get the top 10 brands based on count
        top_brands = sorted(brand_years.keys(), key=lambda x: len(brand_years[x]), reverse=True)[:10]

        num_plots = len(top_brands)
        num_rows = num_plots // 2 + num_plots % 2
        num_cols = 2

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        fig.suptitle("Histogram of Car Production Year per Brand")

        for i, brand in enumerate(top_brands):
            ax = axes[i // num_cols, i % num_cols]

            years = brand_years[brand]
            ax.hist(years, bins=range(int(min(years)), int(max(years)) + 2), alpha=0.7)
            ax.set_title(brand)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

        
    def get_most_common_color(self, collection):
        pipeline = [
            {"$group": {"_id": "$color", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        result = collection.aggregate(pipeline)

        most_common_color = next(result)['_id']

        print("The most common car color is:", most_common_color)


if __name__ == '__main__':
    CarUploader().run()