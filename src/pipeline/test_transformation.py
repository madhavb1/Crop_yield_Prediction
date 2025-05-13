import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation

# Create a small test dataset
def create_test_data():
    data = {
        'Year': [2020, 2021, 2022],
        'CONDITION': [80, 85, 90],
        'PROGRESS': [50, 60, 70],
        'PRICE RECEIVED': [100, 110, 120],
        'YIELD': [200, 210, 220],
        'Period': ['Spring', 'Summer', 'Fall'],
        'State': ['CA', 'TX', 'FL']
    }
    return pd.DataFrame(data)

def test_transformation():
    # Create test data
    df = create_test_data()
    train_path = "test_train.csv"
    test_path = "test_test.csv"
    df.iloc[:2].to_csv(train_path, index=False)
    df.iloc[2:].to_csv(test_path, index=False)
    
    # Test transformation
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
    
    print("Test successful!")
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    
    # Clean up
    import os
    os.remove(train_path)
    os.remove(test_path)

if __name__ == "__main__":
    test_transformation()