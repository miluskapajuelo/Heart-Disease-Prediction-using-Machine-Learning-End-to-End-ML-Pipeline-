
## 1. Classification models
- Logistic Regression: "Uses a weighted combination of features to estimate the probability of heart disease."
- Random Forest: "Uses many decision trees and combines their votes to make a prediction."
- XGBoost: "Builds trees sequentially, with each tree correcting the mistakes of the previous ones."

## 2. Web application 
- focus on customer, nice interface (with extra information)


## 3. ML Pipeline Flowone_hot_encoding
![sandbox](/imagesToLearn/Sandbox.png)


## 4. Concepts
- ML models learn patterns from data
- Use another dataset with real life type, imbalance dataset
- We do not just drop a variable, make sure that combine with others it can have a big impact
- heatmap is just for numerical attibuttes
- Notebooks are not need for development stage, I can use .py
- Understanding system / architecture is important
- Drop the most important attibute for the model, and test how much every feature weight SHAP, Feature importance, trazability.
- Pin al server
- find an open source dataset that you can refresh every day, or downlad everyday
- something that matter for you, in your everyday
- Explanability is very important 

## 5. One-hot encoding 
is a data preprocessing technique that converts categorical text or labels into numerical format, represented as binary vectors, to prevent algorithms from mistakenly assuming an ordinal relationship (e.g., higher numbers are "better" or sequential) between unranked categories.

Use One-Hot Encoding for categorical variables when the model cannot naturally handle categories (Linear Regression, Logistic Regression, SVM, KNN, Neural Networks). Avoid it for high-cardinality features and for algorithms with native categorical support such as CatBoost and LightGBM.

![OHE](/imagesToLearn/one_hot_encoding.jpg)

| ML Model                         | Use One-Hot Encoding? | Notes                                                                            |
| -------------------------------- | --------------------- | -------------------------------------------------------------------------------- |
| **Linear Regression**            | ✅ Yes                 | Required for categorical variables.                                              |
| **Logistic Regression**          | ✅ Yes                 | Standard approach.                                                               |
| **Ridge Regression**             | ✅ Yes                 | Works very well with OHE.                                                        |
| **Lasso Regression**             | ✅ Yes                 | Commonly used with OHE.                                                          |
| **Elastic Net**                  | ✅ Yes                 | Recommended.                                                                     |
| **Support Vector Machine (SVM)** | ✅ Yes                 | Categories must be converted to numbers.                                         |
| **K-Nearest Neighbors (KNN)**    | ✅ Yes                 | OHE prevents artificial ordering.                                                |
| **Neural Networks (MLP)**        | ✅ Yes                 | Usually preferred for low-cardinality features.                                  |
| **Naive Bayes**                  | ✅ Yes                 | Often used for categorical data.                                                 |
| **PCA**                          | ⚠️ Sometimes          | OHE before PCA is possible but may create many sparse dimensions.                |
| **K-Means Clustering**           | ⚠️ Sometimes          | OHE can work but distances become less meaningful with many categories.          |
| **Decision Tree**                | ❌ Usually No          | Can work with label encoding instead.                                            |
| **Random Forest**                | ❌ Usually No          | Trees don't need OHE if categorical handling is available.                       |
| **Gradient Boosting**            | ❌ Usually No          | Prefer ordinal/category support when available.                                  |
| **XGBoost**                      | ⚠️ Depends            | Traditional XGBoost often uses OHE; newer versions support categorical features. |
| **LightGBM**                     | ❌ No                  | Native categorical support is preferred.                                         |
| **CatBoost**                     | ❌ Never               | Designed specifically to handle categorical features directly.                   |
| **Isolation Forest**             | ⚠️ Sometimes          | Depends on preprocessing strategy.                                               |

Do NOT Use One-Hot Encoding
1. High Cardinality Features

Example:

ZIP Code
10001
10002
...

| Model Family              | OHE?                  |
| ------------------------- | --------------------- |
| Linear Models             | ✅ Always              |
| Distance-Based (KNN, SVM) | ✅ Always              |
| Neural Networks           | ✅ Usually             |
| Tree-Based Models         | ⚠️ Usually not needed |
| CatBoost                  | ❌ No                  |
| LightGBM                  | ❌ No                  |
| XGBoost                   | ⚠️ Depends on version |
| Random Forest             | ⚠️ Optional           |



## 6. Statistical Methods (Hypothesis Testing)
Statistical tests provide p-values to determine if the differences in numerical values across categories are statistically meaningful or just random noise

| Scenario                                             | Recommended Statistical Test                                      | Purpose / Measurement                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **2 Categories Only** (e.g., Male vs Female)         | **Two-Sample t-Test**                                             | Compares the means of two independent groups. Assumes a normal distribution.                |
| **3+ Categories** (e.g., Chest Pain Type 1, 2, 3, 4) | **ANOVA (Analysis of Variance)**                                  | Evaluates whether at least one group mean is significantly different from the others.       |
| **Non-Normal Data (Skewed/Outliers)**                | **Mann-Whitney U** (2 groups) <br> **Kruskal-Wallis** (3+ groups) | Non-parametric tests that compare the medians of groups using ranks rather than raw values. |

```python
from scipy import stats

data_df = df.copy()

plt.figure(figsize=(8, 5))
sns.boxplot(data=data_df, x='cp', y='thalach', palette='Set2')
plt.title('Max Heart Rate (thalach) Distribution by Chest Pain Type (cp)')
plt.xlabel('Chest Pain Type (0, 1, 2, 3)')
plt.ylabel('Max Heart Rate Achieved')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

categories = data_df['cp'].unique()
groups = [data_df[data_df['cp'] == cat]['thalach'].values for cat in categories]

# Run the One-Way ANOVA
f_statistic, p_value = stats.f_oneway(*groups)

print(f"ANOVA F-Statistic: {f_statistic:.4f}")
print(f"ANOVA p-value:     {p_value:.4e}")

if p_value < 0.05:
    print("✅ CRITERIA MET: Statistically Significant Relationship!")
    print("The average max heart rate changes significantly depending on the chest pain type.")
    print("Keep 'cp' and 'thalach' as high-priority features for model training.")
else:
    print("❌ NO SIGNIFICANT RELATIONSHIP:")
    print("The differences in heart rate across these categories could be pure random noise.")
```

## 7. Hyperparameter Tuning
is the process of finding the best settings for a machine learning model before training to improve its performance.

use Optuna to search for the best "knobs" of XGBoost — mainly max_depth, learning_rate, n_estimators, subsample, colsample_bytree, and min_child_weight — optimizing ROC-AUC (or recall) on the training data, letting Bayesian optimization focus on the promising regions instead of trying combinations blindly.

## 8. Nested CV  
because if you tune the hyperparameters and measure performance on the same CV, the model indirectly "sees" the validation data and your score comes out optimistic (subtle leakage). Nested CV splits this into two loops — the inner one only for tuning, the outer one only for measuring on data the tuning never touched — giving you an honest estimate of how it would truly generalize. In one sentence: nested CV stops you from fooling yourself when choosing hyperparameters.


## 9. K fold cross validation_test split
![validation](/imagesToLearn/grid_search_cross_validation.png)

## 20.Model in production
monitoring of the model, evaluation, frecuency, flag model drift
update the flow
evaluate the model(determine the frecuency to evaluate)


## 11. Recall
measures the proportion of actual positive cases that the model correctly identifies.

Formula
Recall = TP/(TP + FN)

Where:
TP (True Positives): Positive cases correctly predicted.
FN (False Negatives): Positive cases the model missed.

Example
Suppose there are 33 patients with heart disease. 
The model correctly identifies 26 of them and misses 7.

Recall = 26/(26+7)=0.788
Recall = 78.8%

Interpretation
The model successfully identified 78.8% of patients who actually had heart disease, but it failed to detect the remaining 21.2%.

## 12. UV

- uv init --name heart_disease_prediction_using_machine_learning_end_to_end_ml_pipeline
- uv add -r requirements.txt
- uv run main.py
- reset terminal: source ~/.zshrc

## 13. feature enginnering techniques
In Machine Learning, Feature Engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models.
An ML Engineer sanitizes and structures these transformations based on whether the data is numerical, categorical, text-based, or time-series.
### 1. Numerical Feature Engineering
Continuous variables often need structural adjustments to prevent large numbers from overwhelming your algorithms.
#### 1.a. Feature Scaling (Standardization & Normalization):
StandardScaler: Centers data around a mean of 0 with a standard deviation of 1. Vital for Logistic Regression, SVMs, and Neural Networks.
MinMaxScaler: Scales data strictly between 0 and 1. Useful when you need preserved boundaries (e.g., image processing).
#### 1.b. Log Transformation: 
Applies np.log(x) to highly skewed numerical distributions (like income or cholesterol outliers) to pull data closer to a normal distribution.
#### 1.c. Binning (Discretization): 
Converts continuous data into categorical buckets. 
For example, converting raw age integers into age group intervals (e.g., [0-18, 19-35, 36-60, 60+]).
#### 1.d. Polynomial Features: 
Creates interaction terms by multiplying or squaring features (e.g., creating a new column for age × thalach). 
This helps linear models capture non-linear relationships.

### 2. Categorical Feature Engineering
Machine Learning models only understand math, so text categories must be converted into numerical formats without introducing false assumptions.
#### 1.a. One-Hot Encoding: 
Creates binary columns (0 or 1) for each category. 
Use this for nominal categories without inherent order, like cp (chest pain type) or sex.
#### 1.b. Ordinal Encoding: Converts text categories into ordered integers. 
Use this only when rank matters, such as education level ([High School=1, Bachelors=2, PhD=3]).
#### 1.c. Target Encoding: 
Replaces a categorical string value with the average target value for that specific category. Highly useful for high-cardinality columns (features with hundreds of unique values like ZIP codes).
### 3. Advanced & Context-Specific Techniques
Depending on your data type, you will use specialized transformations to isolate predictive patterns.
#### 1.a. Handling Missing Values (Imputation): 
Filling empty values using statistical markers like the mean, median, or mode, or predicting them using advanced algorithms like KNN Imputer.
#### 1.b. Outlier Handling (Trimming or Winsorization): 
Capping extreme values at a specific percentile (e.g., 1st and 99th percentiles) so that extreme statistical anomalies do not distort your model’s decision boundaries.
#### 1.c. Datetime Transformations: 
Breaking down timestamps into actionable features like hour_of_day, day_of_week, is_weekend, or calculating the duration elapsed between two dates.
#### 1.d. Domain-Specific Extraction: 
Combining medical variables to create indicators like BMI (from height and weight) or generating a "High Cardiac Risk Factor" binary column if thalach is low while oldpeak is high.


## 14. FastAPI for API