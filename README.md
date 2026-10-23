# Loan ML App

A small machine learning web app that works with a loan dataset. It uses two models, served through a FastAPI backend and a simple HTML frontend:

- **Regression** (Linear Regression): predicts the **loan amount** a person is likely to get.
- **Classification** (Logistic Regression): predicts whether a loan will be **Approved or Rejected**, along with a confidence score.

The whole project was built and tested in a Google Colab notebook (`alisha_019.ipynb`), which creates all the files below.

---

## What the notebook does

1. **Creates `requirements.txt` and installs packages**: fastapi, uvicorn, scikit-learn, pandas, joblib, httpx.
2. **Creates project folders** (`data/`, `models/`, `frontend/`) and loads the dataset `cleaned_LOAN.csv`.
3. **Explores the data**: shape, first rows, summary statistics, and class balance.
4. **Trains and saves both models** (`train.py`) and prints their scores.
5. **Writes the FastAPI backend** (`main.py`) that loads the saved models and exposes prediction endpoints.
6. **Writes the web page** (`frontend/index.html`) with two forms, one for each model.
7. **Tests the API** inside the notebook using FastAPI's `TestClient`.
8. **Runs the web app** with uvicorn and shows it inside Colab.

---

## Dataset

File: `data/cleaned_LOAN.csv` (4,269 rows, 8 columns)

| Column | Description |
|---|---|
| `education` | 1 = Graduate, 0 = Not Graduate |
| `self_employed` | 1 = Yes, 0 = No |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term in years |
| `cibil_score` | Credit score (300 to 900) |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | 1 = Approved, 0 = Rejected |

About 62% of the loans in the data are approved and 38% are rejected.

---

## Models

| Task | Model | Target | Features | Result |
|---|---|---|---|---|
| Regression | `LinearRegression` | `loan_amount` | education, self_employed, income_annum, loan_term, cibil_score, bank_asset_value | R² = **0.8513** |
| Classification | `StandardScaler` + `LogisticRegression` | `loan_status` | all of the above plus `loan_amount` | Accuracy = **0.9122** |

Both use an 80/20 train/test split with `random_state=42`. The classification split is stratified. Scaling is used for the classifier because income and loan values are in the millions while the CIBIL score is only 300 to 900.

Trained models are saved with joblib in the `models/` folder.

---

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serves the web page |
| POST | `/predict/regression` | Returns `predicted_loan_amount` |
| POST | `/predict/classification` | Returns `loan_status`, `label`, and `confidence` |

Example request to `/predict/classification`:

```json
{
  "education": 1,
  "self_employed": 0,
  "income_annum": 5000000,
  "loan_amount": 15000000,
  "loan_term": 10,
  "cibil_score": 650,
  "bank_asset_value": 5000000
}
```

Example response:

```json
{ "loan_status": 1, "label": "Approved", "confidence": 0.9576 }
```

Inputs are validated with Pydantic (for example, CIBIL score must be between 300 and 900).

---

## Project structure

```
loan-ml-app/
├── data/
│   └── cleaned_LOAN.csv
├── models/
│   ├── regression_model.joblib
│   └── classification_model.joblib
├── frontend/
│   └── index.html
├── main.py              # FastAPI backend
├── train.py             # trains and saves both models
├── requirements.txt
├── alisha_019.ipynb     # the original notebook
└── README.md
```

---

## How to run locally

1. Clone the repository and open the folder:
   ```bash
   git clone https://github.com/YOUR-USERNAME/loan-ml-app.git
   cd loan-ml-app
   ```
2. (Optional) create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the models (creates the `models/` files):
   ```bash
   python train.py
   ```
5. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
6. Open **http://127.0.0.1:8000** in your browser.

---

## Notes

- The model is for learning and demonstration only. It should not be used to make real lending decisions.
- Classification confidence is the probability of the predicted class.
- The frontend works when served by FastAPI, or when `index.html` is opened directly (it then talks to `http://127.0.0.1:8000`).

## Tech stack

Python, pandas, scikit-learn, joblib, FastAPI, Uvicorn, HTML/CSS/JavaScript
