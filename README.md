# Concept Drift Detection MLOps Pipeline(Ongoing)

An end-to-end MLOps pipeline for detecting concept drift in machine learning models with automatic retraining capabilities. This project demonstrates a complete workflow from model training to deployment with drift monitoring.

##  Project Overview

This pipeline monitors machine learning model performance in production and detects when the underlying data distribution changes (concept drift). When drift is detected, the system automatically retrains the model on new data to maintain performance.

### Key Features

- **Automated Concept Drift Detection** using Kolmogorov-Smirnov (KS) statistical test
- **Model Retraining Pipeline** triggered when drift is detected
- **FastAPI REST API** for model inference
- **Airline Customer Satisfaction** dataset as use case
- **Random Forest Classifier** as the base model
- **Data Preprocessing** with one-hot encoding and missing value handling

##  Dataset

The project uses the Airline Customer Satisfaction dataset with the following features:

- **Customer demographics**: Age, Gender, Customer Type
- **Travel details**: Type of Travel, Class, Flight Distance
- **Service ratings**: Inflight wifi, Food and drink, Seat comfort, etc.
- **Delay information**: Departure/Arrival delays
- **Target variable**: Customer satisfaction (satisfied/neutral or dissatisfied)

##  Architecture

```
┌─────────────────┐
│   Training Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │ → model.pkl
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │ → /predict endpoint
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ New Data Stream │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Drift Detection │ ← KS Test
└────────┬────────┘
         │
    Drift?
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
    ▼         │
┌─────────┐   │
│Retraining│   │
└────┬────┘   │
     │       │
     ▼       │
┌─────────┐   │
│model_v2 │   │
└────┬────┘   │
     │       │
     └───┬───┘
         ▼
┌─────────────────┐
│  Updated Model  │
└─────────────────┘
```

##  Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Jess33B/Concept-Drift-Detection-MLOps-Pipeline.git
cd Concept-Drift-Detection-MLOps-Pipeline
```

2. Install dependencies:
```bash
pip install fastapi uvicorn pandas scikit-learn scipy pickle5
```

### Usage

#### 1. Train the Initial Model

Run the notebook `Untitled-1.ipynb` to:
- Load and preprocess the training data
- Train the Random Forest model
- Save the model as `model.pkl`
- Save training features as `X_train.csv`

#### 2. Start the API Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

#### 3. Make Predictions

Send a POST request to the `/predict` endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Flight Distance": 500,
    "Inflight wifi service": 3,
    "Departure/Arrival time convenient": 4,
    "Ease of Online booking": 3,
    "Gate location": 3,
    "Food and drink": 4,
    "Online boarding": 3,
    "Seat comfort": 4,
    "Inflight entertainment": 4,
    "On-board service": 4,
    "Leg room service": 3,
    "Baggage handling": 4,
    "Checkin service": 3,
    "Inflight service": 4,
    "Cleanliness": 4,
    "Departure Delay in Minutes": 10,
    "Arrival Delay in Minutes": 5,
    "Gender_Male": 1,
    "Customer Type_disloyal Customer": 0,
    "Type of Travel_Personal Travel": 0,
    "Class_Eco": 1,
    "Class_Eco Plus": 0
  }'
```

Response:
```json
{
  "prediction": 1
}
```

##  Concept Drift Detection

### How It Works

The pipeline uses the **Kolmogorov-Smirnov (KS) test** to detect concept drift:

1. Compare the distribution of each feature in the training data vs. new data
2. Calculate p-value for each feature using KS test
3. If p-value < 0.05, the feature has drifted significantly
4. If any features drift, trigger model retraining

### Example

```python
from scipy.stats import ks_2samp

# Compare training data with new data
stat, p_value = ks_2samp(X_train[col], X_new[col])

if p_value < 0.05:
    print(f"Drift detected in {col}")
```

### Simulated Drift

The project includes `drifted_data.csv` which simulates concept drift by:
- Increasing flight distance by 50%
- Increasing age by 15 years
- Adding 50 minutes to departure delays

This demonstrates the drift detection system in action.

## Project Structure

```
.
├── app.py                      # FastAPI application
├── Untitled-1.ipynb           # Jupyter notebook with pipeline code
├── model.pkl                  # Initial trained model
├── model_v2.pkl              # Retrained model after drift
├── X_train.csv               # Training features
├── drifted_data.csv          # Simulated drifted data
└── README.md                 # This file
```

##  Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: Airline customer satisfaction dataset
- **Features**: 23 features (demographics, service ratings, delays)
- **Target**: Binary classification (satisfied vs. neutral/dissatisfied)
- **Performance**: Baseline accuracy on training data

## MLOps Pipeline Stages

1. **Data Ingestion**: Load training and test datasets
2. **Preprocessing**: Handle missing values, encode categorical variables
3. **Model Training**: Train Random Forest on training data
4. **Model Serialization**: Save model as pickle file
5. **API Deployment**: Serve model via FastAPI
6. **Drift Monitoring**: Continuously monitor incoming data
7. **Drift Detection**: Statistical tests to detect distribution changes
8. **Model Retraining**: Automatic retraining when drift detected
9. **Model Update**: Deploy new model version

##  Continuous Improvement

The pipeline supports continuous model improvement through:

- **Automated drift detection** without manual intervention
- **Automatic retraining** when data distribution changes
- **Model versioning** (model.pkl, model_v2.pkl, etc.)
- **API hot-reload** for seamless model updates

##  Technologies Used

- **Python**: Core programming language
- **FastAPI**: Web framework for API
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning library
- **SciPy**: Statistical tests (KS test)
- **Pickle**: Model serialization




