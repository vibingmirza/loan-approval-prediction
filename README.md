# Loan Approval Prediction

## 1. Project Overview

This project predicts whether a loan application will be approved or rejected using Machine Learning.

The problem is a **binary classification problem** where:

- 1 = Loan Approved
- 0 = Loan Rejected

The model used in this project is **Logistic Regression**.

## 2. Dataset

The project uses a Loan Prediction dataset containing **614 loan applications** and **13 original columns**.

The dataset contains information such as:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status

`Loan_ID` was removed because it is only an identifier and does not help the model make predictions.

## 3. Data Preprocessing

The following preprocessing steps were performed:

1. Missing categorical values were filled using the mode.
2. Missing numerical values were filled using the median.
3. The `3+` value in Dependents was converted to `3`.
4. `Loan_ID` was removed.
5. Categorical variables were converted into numerical values using One-Hot Encoding.
6. The target variable `Loan_Status` was converted into:
   - 1 = Approved
   - 0 = Rejected

After preprocessing, the dataset contained:

- 614 rows
- 15 columns
- 0 missing values
- 0 duplicate records

## 4. Exploratory Data Analysis

Several visualizations were created to understand the dataset:

- Loan Approval Distribution
- Credit History vs Loan Approval
- Applicant Income vs Loan Approval
- Education vs Loan Approval

The analysis showed that **Credit History has a strong relationship with loan approval**.

The dataset contains:

- 422 approved applications
- 192 rejected applications

## 5. Machine Learning Model

The machine learning model used is **Logistic Regression**.

The dataset was divided into:

- 80% training data
- 20% testing data

The model was trained using the training dataset and then used to predict loan approval for unseen test data.

## 6. Model Evaluation

The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

These metrics were used to measure how well the model predicts loan approval and rejection.

## 7. Loan Prediction Application

A simple interactive application was created using Python and `ipywidgets`.

Users can enter:

- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area

The application provides:

- Loan approval probability
- Approval or rejection prediction

Example test result:

**Approval Probability: 83.06%**

**Prediction: APPROVED**

## 8. Project Screenshots

Screenshots of the following are included as project proof:

1. Loan Prediction Application UI
2. Applicant Information
3. Approval Probability
4. Approval/Rejection Prediction
5. Model Evaluation
6. Multiple Test Cases

## 9. Limitations

This project has some limitations:

- The dataset is relatively small.
- The model depends on the quality of the available dataset.
- Real-world banks may use many additional financial and customer factors.
- The prediction should not be considered a final financial decision.
- Logistic Regression may not capture complex relationships between all variables.

## 10. Conclusion

The Loan Approval Prediction project demonstrates how Machine Learning can be used to predict loan approval based on applicant information.

Logistic Regression was trained and evaluated using a real loan prediction dataset. An interactive prediction application was also developed to allow users to enter applicant information and receive an approval probability and prediction.

This project demonstrates the complete Machine Learning workflow from data preprocessing and exploration to model training, evaluation, and application development.