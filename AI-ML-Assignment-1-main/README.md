# Medical Cost Insurance Prediction

## Objective
The primary objective of this project is to build a **Multiple Linear Regression** machine learning model to estimate individual medical insurance charges based on personal and health-related factors. By analyzing demographic and health data, the model helps identify key drivers of medical costs for insurance underwriting.

## Dataset Link
* **Source:** [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* **Target Variable:** `charges` (continuous numerical value representing individual medical costs)
* **Features:**
  * **Numerical:** `age`, `bmi`, `children`
  * **Categorical:** `sex`, `smoker`, `region`

## Libraries Used
* **`pandas`**: Data manipulation and feature extraction
* **`numpy`**: Numerical arrays and calculations
* **`scikit-learn`**: Preprocessing (`train_test_split`), model building (`LinearRegression`), and evaluation metrics (`MAE`, `MSE`, `R2 Score`)
* **`matplotlib`**: Plotting the actual vs. predicted charges scatter plot
* **`seaborn`**: Statistical data visual styles and enhancement

## Methodology

### 1. Data Understanding & Preprocessing
* Checked for missing or null values across all columns.
* Converted categorical variables (`sex`, `smoker`, `region`) into numerical features using **One-Hot Encoding** (`pd.get_dummies`) with `drop_first=True` to avoid the dummy variable trap (multicollinearity).
* Split the dataset into **80% training data** (for model learning) and **20% testing data** (for evaluation).

### 2. Model Development
* Built a **Multiple Linear Regression** model trained on features: `age`, `bmi`, `children`, and encoded columns for `sex`, `smoker`, `region`.
* Fitted the model on the training set and generated predictions on the unseen test set.

### 3. Model Evaluation
* Measured overall predictive accuracy using standard regression metrics:
  * Mean Absolute Error (**MAE**)
  * Mean Squared Error (**MSE**)
  * Coefficient of Determination (**$R^2$ Score**)
* Visualized performance using an **Actual vs. Predicted Charges** scatter plot.

## Results

* **$R^2$ Score:** ~0.783
* **Mean Absolute Error (MAE):** ~$4,180
* **Root Mean Squared Error (RMSE):** ~$5,790

### Key Observations
* The model explains approximately **78.3% of the variance** in medical costs across test samples.
* Feature coefficients indicate that **smoking status** is the single largest predictor of higher insurance charges, followed by **age** and **BMI**.
* Scatter plots reveal distinct clusters, indicating non-linear patterns that standard linear regression struggles to capture fully.

## Conclusion
The multiple linear regression model successfully identifies the primary drivers of medical insurance costs the most crucial finding is that a persons smoking status dictates the highest increase in medical charges overwhelming all other variables demographic details like region and gender show minimal impact compared to core health metrics both age and body mass index steadily drive up expected costs as they increase one major limitation of applying linear regression to this problem is its strict assumption of independent linear relationships the model simply adds the independent effects of features together meaning it fails to recognize complex overlapping interactions higher body mass index causes a massive surge in medical costs almost exclusively for people who smoke but a standard linear model cannot naturally detect this combined effect without manually creating new interaction features.
