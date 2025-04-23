# NYC Taxi ML Pipeline

## 🚀 Project Overview
This project implements a **Machine Learning Pipeline** for predicting NYC taxi fares using structured data from taxi trips. The pipeline covers **data ingestion, preprocessing, feature engineering, model training, and inference** using **LightGBM** for efficient training.

## 📂 Project Structure
```
NYC_Taxi_ML_Pipeline/
│-- main.py                # Main script to run the pipeline
│-- setup_folders.py       # Initializes necessary folders
│-- requirements.txt       # Required dependencies
│-- .gitignore             # Ignore unnecessary files
│
├── notebooks/             # Jupyter notebooks for EDA and experimentation
│   ├── EDA.ipynb
│
├── src/                   # Core pipeline scripts
│   ├── data_ingestion.py   # Loads taxi data
│   ├── data_cleaning.py    # Filters and cleans data
│   ├── feature_engineering.py  # Extracts useful features
│   ├── preprocessing.py    # Prepares data for ML models
│   ├── model_training.py   # Trains LightGBM with early stopping
│   ├── inference.py        # Generates predictions for unseen data
│   ├── config.py           # Configuration parameters
│
├── utils/                 # Helper functions
│   ├── api_utils.py        # Fetches external data (e.g., weather API)
│   ├── visualizations.py   # Custom plotting functions
```

## 🛠 Installation & Setup
### 1. Set up a virtual environment
```sh
python -m venv nyc_taxi_env
source nyc_taxi_env/bin/activate  # macOS/Linux
nyc_taxi_env\Scripts\activate    # Windows
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Obtain API Key from Visual Crossing
This project requires a weather API key from **Visual Crossing**. Note that the free tier only allows you to access 1000 record per day.
**Steps to get an API Key:**
1. Go to [Visual Crossing Weather](https://www.visualcrossing.com/).
2. Sign up for a free account (or log in).
3. Navigate to "**API Keys**" in your account settings.
4. Copy your API key.

### 4. Initialize the Project
Once you have your API key, run the following command to set up required folders and environment files:
```sh
python initialize_project.py
```

### 5. Add API Key to `.env`
Open the `.env` file in a text editor and replace `None` with your actual API key:
```ini
API_KEY=your-api-key-here
```

### 6. Download Yellow Taxi Data and Taxi Zone Lookup Table
Navigate to the [TLC Trip Record Data website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and download:
- **January 2024 Yellow Taxi Trip Records**
- **Taxi Zone Lookup Table**

🔹 **Move the downloaded files into the `data/raw/` directory.**

You may also download additional months of trip records if you want to train the model on a different dataset or make predictions for other months.

⚠️ Temporary Method: This manual download step will be replaced with automated data retrieval in future updates. (See Planned Future Improvements for details.)

## 🚀 Running the Pipeline
### 1. Run Data Processing and Model Training
```sh
python main.py
```

### 2. Run Inference on New Data
```sh
python src/inference.py --input_file "data/raw/yellow_tripdata_2024-02.parquet" --output_file "data/predictions/predictions_2024-02.csv"
```

## 3. Using the FastAPI Inference API

This project includes a REST API for real-time predictions via FastAPI.

### 🔧 Running the API Locally
From the project root directory, run:

```bash
uvicorn src.app:app --reload
```

Then visit:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc UI: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 🧪 Example API Request (Swagger or `curl`)
You can send a request to `/predict` with a payload like:

```json
{
  "data": [
    {
      "VendorID": 2,
      "passenger_count": 1,
      "trip_distance": 2.5,
      "RatecodeID": 1,
      "store_and_fwd_flag": "N",
      "PULocationID": 186,
      "DOLocationID": 236,
      "payment_type": 1,
      "tpep_pickup_datetime": 1708543608,
      "tpep_dropoff_datetime": 1708544008,
      "total_amount": 15.0
    }
  ]
}
```

You’ll receive a response like:

```json
{
  "predictions": [14.72]
}
```

### 🧰 Optional: Test with Python
```python
import requests

payload = {
    "data": [
        {
            "VendorID": 2,
            "passenger_count": 1,
            "trip_distance": 2.5,
            "RatecodeID": 1,
            "store_and_fwd_flag": "N",
            "PULocationID": 186,
            "DOLocationID": 236,
            "payment_type": 1,
            "tpep_pickup_datetime": 1708543608,
            "tpep_dropoff_datetime": 1708544008,
            "total_amount": 15.0
        }
    ]
}

res = requests.post("http://localhost:8000/predict", json=payload)
print(res.json())
```

## 📊 Model Performance
- The model is evaluated using **RMSE, MAE, and R²**.

## 🛠 Planned Future Improvements
### **Automating Data Retrieval**
- Convert manual downloads into an **automated data ingestion pipeline**.
- Fetch taxi data directly from the **AWS S3 bucket** instead of downloading from the website.

### **Deploying the Model via AWS**
- Implement a **Lambda function** to deploy the model.
- Use **API Gateway** to expose an endpoint for predictions.
- Package LightGBM dependencies for **AWS Lambda compatibility**.

### **Adding a Web Interface**
- Create a simple UI for **submitting requests & viewing predictions**.
- Host it via **GitHub Pages** or **AWS S3 static hosting**.
- Implement:
  - **Pickup/dropoff selection** via Google Maps API.
  - **TLC Taxi Zone mapping** using `Geopandas` for geospatial lookups.

### **Enhancing API Key Management**
- Develop an **API Key Request Lambda** to generate temporary API keys.
- Store keys in **DynamoDB** with usage limits.
- Allow users to request & track **API key expiration**.

### **Improving Model Performance**
- Implement **hyperparameter tuning** with **Optuna**.
- Expand training dataset beyond **January 2024** for better generalization.
- Investigate **automated model retraining** based on new data.

---

## 🔬 Other Possible Enhancements
- Analyze **feature importance** using **SHAP values**.
- Test additional ML models (**XGBoost, CatBoost, etc.**).
- Implement **data drift detection** to monitor prediction accuracy.
- Optimize big data processing using **Dask** or **Vaex**.
- Factor in **major NYC events** (e.g., marathons, concerts) affecting taxi demand.

---
## Notes
### **Automated Data Retrieval**
There is no easily accessilbe S3 bucket for the data I would like to work with, so I attempted to use API but the data is much too large for this to be a feasible option. I may come back around to this later.

---
🎯 **Developed for NYC Taxi Fare Prediction with Scalable ML Practices!**

