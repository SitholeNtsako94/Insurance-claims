# 📊 Insurance Claim Classification

A Streamlit app that uses Logistic Regression to predict whether an insurance policy will result in a claim. It trains on historical insurance claims data and shows how well the model performs on records it has never seen.

> **Note:** This is a demonstration project. It does not accept manually entered policy details and should not be used for real underwriting or pricing decisions.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)

## Features

- **Dataset overview:** record counts, number of features, a data preview, and a chart of the claim status distribution.
- **Model evaluation:** accuracy, claim precision and recall, a confusion matrix, and a full classification report.
- **Actual vs. predicted results:** a side-by-side comparison of the model's predictions and the true outcomes on the test set.

## How It Works

The target variable is `claim_status` (`0` = no claim, `1` = claim).

1. **Load and clean:** rows with a missing or invalid `claim_status` are removed.
2. **Select features:** `claim_status` and `policy_id` are dropped so the model cannot learn from the target or from an identifier.
3. **Split:** 80% training and 20% testing, stratified so both sets keep the same claim ratio (`random_state=42`).
4. **Preprocess** with a scikit-learn pipeline:
   - Numeric columns: median imputation, then standard scaling.
   - Categorical columns: most-frequent imputation, then one-hot encoding (unknown categories are ignored).
5. **Train:** Logistic Regression with `class_weight="balanced"` to account for the imbalance between claims and non-claims.
6. **Evaluate:** metrics are computed on the held-out test set only.

The model is trained once on startup and cached with `st.cache_resource`, so the app doesn't retrain on every interaction.

## Project Structure

```
.
├── app.py                      # Streamlit app
├── importDataset.ipynb         # Notebook used for data exploration
├── Insurance claims data.csv   # Dataset (must sit next to app.py)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or newer

### Installation

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install streamlit pandas scikit-learn matplotlib
```

### Add the Dataset

`app.py` reads `Insurance claims data.csv` (with spaces) from the same folder. If your file is named `Insurance_claims_data.csv`, rename it to match, or update the filename in `app.py`.

The CSV must contain a `claim_status` column with values `0` and `1`.

### Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually http://localhost:8501).

## Tech Stack

- [Python](https://www.python.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)

## Troubleshooting

| Problem | Likely cause and fix |
| --- | --- |
| "Could not load the data or train the model" | The CSV is missing, misnamed, or not in the same folder as `app.py`. |
| Error mentioning `claim_status` | The dataset has no `claim_status` column, or it holds values other than 0 and 1. |

## Possible Improvements

- Compare against other models (Random Forest, Gradient Boosting).
- Add cross-validation and threshold tuning.
- Add a form for predicting a single policy from manually entered details.
- Show feature importance or coefficients to explain predictions.
