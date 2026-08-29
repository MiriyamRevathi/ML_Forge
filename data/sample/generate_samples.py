"""
MLForge - Benchmark Sample Dataset Generator
Creates local CSV datasets for instant execution of ML pipelines:
1. Customer Churn (Classification)
2. House Prices (Regression)
3. Iris Flowers (Multi-class Classification)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path to access config
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import SAMPLE_DIR, DATASET_DIR
from utils.files import save_json
from utils.helpers import get_current_timestamp_iso


def generate_customer_churn_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generates synthetic customer churn binary classification dataset."""
    np.random.seed(42)
    
    age = np.random.randint(18, 70, size=n_samples)
    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)
    total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 10, size=n_samples), 2)
    total_charges = np.maximum(total_charges, 0)
    
    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.5, 0.3, 0.2])
    payment_methods = np.random.choice(["Electronic Check", "Mailed Check", "Credit Card", "Bank Transfer"], size=n_samples)
    internet_service = np.random.choice(["Fiber Optic", "DSL", "No"], size=n_samples, p=[0.4, 0.4, 0.2])
    tech_support = np.random.choice(["Yes", "No"], size=n_samples, p=[0.4, 0.6])
    
    # Calculate churn probability based on feature rules
    logit = (
        0.03 * (60 - tenure_months) +
        0.02 * monthly_charges +
        (1.2 if "Month-to-Month" in contract_types else -0.8) +
        (0.8 if "Fiber Optic" in internet_service else -0.4) -
        (0.7 if "Yes" in tech_support else 0.0) -
        2.5
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i+1000}" for i in range(n_samples)],
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "churn": churn
    })
    
    # Introduce a few realistic missing values for testing data cleaning
    missing_indices = np.random.choice(n_samples, size=15, replace=False)
    df.loc[missing_indices[:8], "total_charges"] = np.nan
    df.loc[missing_indices[8:], "tech_support"] = np.nan
    
    return df


def generate_house_prices_dataset(n_samples: int = 500) -> pd.DataFrame:
    """Generates synthetic house prices regression dataset."""
    np.random.seed(42)
    
    square_feet = np.random.randint(600, 4500, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=n_samples)
    year_built = np.random.randint(1950, 2024, size=n_samples)
    garage_cars = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.1, 0.4, 0.4, 0.1])
    neighborhood = np.random.choice(["Downtown", "Suburbs", "Highlands", "Waterfront"], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    condition = np.random.choice(["Fair", "Good", "Excellent"], size=n_samples, p=[0.2, 0.6, 0.2])
    
    # Price formula
    neighborhood_factor = {"Downtown": 45000, "Suburbs": 20000, "Highlands": 65000, "Waterfront": 120000}
    condition_factor = {"Fair": -15000, "Good": 0, "Excellent": 35000}
    
    n_boost = np.array([neighborhood_factor[n] for n in neighborhood])
    c_boost = np.array([condition_factor[c] for c in condition])
    
    price = (
        120 * square_feet +
        15000 * bedrooms +
        12000 * bathrooms +
        800 * (year_built - 1950) +
        18000 * garage_cars +
        n_boost +
        c_boost +
        np.random.normal(0, 25000, size=n_samples)
    )
    price = np.round(np.maximum(price, 50000), 2)
    
    df = pd.DataFrame({
        "house_id": [f"HOUSE_{i+100}" for i in range(n_samples)],
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "garage_cars": garage_cars,
        "neighborhood": neighborhood,
        "condition": condition,
        "price": price
    })
    
    return df


def generate_iris_dataset() -> pd.DataFrame:
    """Generates standard Iris flower multi-class classification dataset."""
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species_code"]
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = df["species_code"].map(species_map)
    df = df.drop(columns=["species_code"])
    return df


def main():
    print("Generating MLForge benchmark sample datasets...")
    
    datasets = [
        ("customer_churn.csv", generate_customer_churn_dataset(500), "churn", "classification", "Customer Churn Prediction Dataset"),
        ("house_prices.csv", generate_house_prices_dataset(500), "price", "regression", "House Prices Regression Dataset"),
        ("iris.csv", generate_iris_dataset(), "species", "classification", "Iris Flower Classification Benchmark Dataset")
    ]
    
    for filename, df, target_col, task_type, desc in datasets:
        # Save to sample directory
        sample_path = SAMPLE_DIR / filename
        df.to_csv(sample_path, index=False)
        print(f"Saved sample CSV: {sample_path}")
        
        # Copy to primary datasets directory with metadata JSON
        dataset_path = DATASET_DIR / filename
        df.to_csv(dataset_path, index=False)
        
        meta_id = filename.rsplit('.', 1)[0]
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        metadata = {
            "id": meta_id,
            "filename": filename,
            "name": desc,
            "target_column": target_col,
            "task_type": task_type,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_values_count": int(df.isna().sum().sum()),
            "file_size": f"{sample_path.stat().st_size / 1024:.2f} KB",
            "created_at": get_current_timestamp_iso(),
            "is_sample": True
        }
        
        meta_path = DATASET_DIR / f"{meta_id}_meta.json"
        save_json(metadata, meta_path)
        print(f"Saved dataset metadata JSON: {meta_path}")

    print("Sample datasets successfully created!")


if __name__ == "__main__":
    main()
