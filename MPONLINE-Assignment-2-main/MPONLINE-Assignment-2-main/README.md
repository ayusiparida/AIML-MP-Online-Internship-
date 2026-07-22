# Customer Churn Prediction using Logistic Regression

**Author:** AYUSI PARIDA

**Registration Number:** 23Bhi10004

**Application Number:** IN26011773

**Batch Number:** 1A

**Email ID:** ayusi.23bhi10004@vitbhopal.ac.in

## Objective
The objective of this project is to build a Logistic Regression model to predict customer churn for a telecommunications company based on customer demographics and service usage patterns.

## Dataset Link
- [Kaggle: Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kaggle`

## Methodology
1. **Data Understanding**: Loaded the dataset and categorized features into numerical and categorical components while isolating `Churn` as the target variable.
2. **Data Preprocessing**:
   - Cleaned `TotalCharges` by converting to numeric and imputing missing values with the median.
   - Encoded `Churn` target as binary (`Yes`: 1, `No`: 0).
   - Generated dummy variables for categorical features using One-Hot Encoding (`drop_first=True`).
   - Standardized features using `StandardScaler` to ensure convergence during optimization.
3. **Data Splitting**: Divided the dataset into 80% training set and 20% testing set using `train_test_split`.
4. **Model Development**: Trained a `LogisticRegression` classifier with `max_iter=1000` on scaled features.
5. **Model Evaluation**: Evaluated model performance using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix heatmap.

## Results
- **Accuracy:** 81.97%
- **Precision:** 68.31%
- **Recall:** 59.52%
- **F1-Score:** 0.6361

## Conclusion
Logistic Regression effectively models customer churn with an overall accuracy of 81.97%. Contract type, tenure length, and monthly charges serve as significant predictors. The primary limitation of Logistic Regression is its linear decision boundary, which misses complex non-linear feature interactions (such as high monthly charges combined with short tenure). Non-linear models like Random Forest or XGBoost would likely yield better recall for detecting at-risk customers.
