import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Page configuration
st.set_page_config(
    page_title="Insurance Claim Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Insurance Claim Classification")
st.write(
    "A machine learning demonstration using insurance claims data "
    "and Logistic Regression."
)

# Load and train the model once, then cache the result
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Insurance claims data.csv")

    # Prepare the target variable
    df = df.dropna(subset=["claim_status"]).copy()
    y = pd.to_numeric(df["claim_status"], errors="coerce")

    valid_rows = y.isin([0, 1])
    df = df.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    # Remove target and identifier from the features
    X = df.drop(
        columns=["claim_status", "policy_id"],
        errors="ignore"
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Identify column types
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Preprocess numeric features
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocess categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Create and train the model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred = model.predict(X_test)

    return df, X, y_test, y_pred, model


# Train and evaluate, showing a helpful error if something fails
try:
    with st.spinner("Loading data and training the model..."):
        df, X, y_test, y_pred, model = load_and_train_model()
except Exception as e:
    st.error(f"Could not load the data or train the model: {e}")
    st.info(
        "Check that the CSV file is in the same folder as app.py "
        "and that it contains a claim_status column."
    )
    st.stop()

st.success("Model trained successfully!")

# Dataset overview
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total records", f"{len(df):,}")
col2.metric("Features used", X.shape[1])
col3.metric("Test records", f"{len(y_test):,}")

st.write("Preview of the dataset")
st.dataframe(df.head(10), use_container_width=True)

# Class distribution
st.subheader("Claim Status Distribution")

class_counts = (
    df["claim_status"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

distribution = pd.DataFrame({
    "Outcome": ["No Claim (0)", "Claim (1)"],
    "Number of records": class_counts.values
})

st.bar_chart(
    distribution.set_index("Outcome")
)

# Model evaluation
st.header("2. Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Claim", "Claim"],
    output_dict=True,
    zero_division=0
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Accuracy", f"{accuracy:.2%}")
metric2.metric("Claim precision", f"{report['Claim']['precision']:.2%}")
metric3.metric("Claim recall", f"{report['Claim']['recall']:.2%}")

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["No Claim (0)", "Claim (1)"],
    cmap="Blues",
    ax=ax
)

ax.set_title("Insurance Claim Classification")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()
st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# Actual versus predicted results
st.header("3. Actual vs Predicted Results")

comparison = pd.DataFrame({
    "Actual claim_status": y_test,
    "Predicted claim_status": y_pred
})

comparison["Result"] = (
    comparison["Actual claim_status"]
    == comparison["Predicted claim_status"]
).map({
    True: "Correct",
    False: "Incorrect"
})

st.write("Compare the model's predictions with the actual outcomes.")
st.dataframe(comparison.head(10), use_container_width=True)

st.caption(
    "Evaluation uses a held-out test set. This demonstration "
    "does not accept manually entered insurance details."
)
