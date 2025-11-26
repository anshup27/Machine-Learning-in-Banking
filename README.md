# Machine-Learning-in-Banking
Project Title: Predicting Bank Customer Loan Default using Machine Learning, Problem Statement: A bank wants to build a machine learning model that can predict whether a customer will default on a loan based on their financial, demographic, and transaction details.
Loan Default Prediction
Overview
A machine learning pipeline to predict loan default using Logistic Regression, Random Forest,
and XGBoost.
Includes preprocessing, feature engineering, model training, and evaluation.
Untitled document
Dataset
Target column: Default
Loaded from:
df = pd.read_csv("Loan_default.csv")
Preprocessing
Identify columns:
numeric_cols = X.select_dtypes(include=['int64','float64'])
categorical_cols = X.select_dtypes(include=['object'])
Pipelines:
● Numeric → median + scaling
● Categorical → most_frequent + one-hot
Combined using ColumnTransformer.
Untitled document
Feature Engineering
X["Income_to_Loan"] = X["Income"] / (X["LoanAmount"] + 1)
X["CreditLines_per_Year"] = X["NumCreditLines"] /
((X["MonthsEmployed"]/12) + 1)
Models
models = {
"Logistic Regression": LogisticRegression(max_iter=200),
"Random Forest": RandomForestClassifier(n_estimators=100,
max_depth=10),
"XGBoost": XGBClassifier(n_estimators=80, max_depth=4, subsample=0.8)
}
Each model is wrapped in a pipeline and trained.
Evaluation
Metrics computed:
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
Plot ROC for all models:
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
XGBoost achieved the highest AUC.
(ROC curve shown in PDF, page 4.)
Untitled document
Confusion Matrices
Generated per model:
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
(Shown on pages 6–8.)
Untitled document
Best Model
Sorted by ROC-AUC:
best = results_df.sort_values("ROC-AUC", ascending=False).iloc[0]
XGBoost performs best
