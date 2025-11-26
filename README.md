# Machine-Learning-in-Banking
Project Title: Predicting Bank Customer Loan Default using Machine Learning, Problem Statement: A bank wants to build a machine learning model that can predict whether a customer will default on a loan based on their financial, demographic, and transaction details.
Loan Default Prediction — Machine Learning Pipeline

A complete machine learning pipeline to predict Loan Default using Logistic Regression, Random Forest, and XGBoost.
Includes data preprocessing, feature engineering, model training, and evaluation with visualizations.

 Dataset

Target Column: Default

Loaded using:

df = pd.read_csv("Loan_default.csv")

 Preprocessing
 Identify Feature Types
numeric_cols = X.select_dtypes(include=['int64', 'float64'])
categorical_cols = X.select_dtypes(include=['object'])

 Pipelines Used
Feature Type	Preprocessing Steps
Numeric Features	Median Imputation → Standard Scaling
Categorical	Most Frequent Imputation → One-Hot Encode

Implemented using ColumnTransformer.

 Feature Engineering
X["Income_to_Loan"] = X["Income"] / (X["LoanAmount"] + 1)

X["CreditLines_per_Year"] = X["NumCreditLines"] / (
    (X["MonthsEmployed"] / 12) + 1
)


These engineered features enhance the model’s ability to capture customer risk ratios.

 Models Used
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10
    ),
    "XGBoost": XGBClassifier(
        n_estimators=80, max_depth=4, subsample=0.8
    )
}


Each model is wrapped into a Pipeline which includes all preprocessing steps.

 Evaluation Metrics

Computed for each model:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)


📌 XGBoost achieved the highest ROC-AUC score.

 Confusion Matrices

Generated for each model:

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


(Originally shown in the PDF, pages 6–8.)

 Best Model

Sorted by ROC-AUC:

best = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]

 Final Winner: XGBoost

XGBoost performed best across most evaluation metrics, especially ROC-AUC.

 Project Structure (Recommended)
 Loan-Default-Prediction
│── README.md
│── Loan_default.csv
│── model_training.ipynb
│── requirements.txt
│── plots/
│   ├── roc_curve.png
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_rf.png
│   └── confusion_matrix_xgb.png

 How to Run

Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook model_training.ipynb
