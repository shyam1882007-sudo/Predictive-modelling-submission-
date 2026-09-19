THIRANEX SUBMISSION
Predictive Modeling Using Machine Learning

Project Title:
Breast Cancer Diagnosis Prediction Using Random Forest

Objective:
Build a supervised machine learning model that predicts whether a breast tumor is
malignant or benign from diagnostic measurements.

Dataset:
Breast Cancer Wisconsin Diagnostic Dataset available through scikit-learn.
Samples: 569
Features: 30

Method:
1. Load the dataset.
2. Separate features and target.
3. Split data into 80% training and 20% testing using stratification.
4. Train a Random Forest Classifier with 200 trees.
5. Generate predictions.
6. Evaluate using accuracy, classification report, confusion matrix and ROC-AUC.

Results:
Accuracy: 0.9561 (95.61%)
ROC-AUC: 0.9931

Files:
- predictive_modeling.py : complete source code
- breast_cancer_dataset.csv : dataset used
- confusion_matrix.png : model performance visualization
- roc_curve.png : ROC visualization
- Thiranex_ML_Report.pdf : formatted project report

Note:
This is an educational machine-learning project. The model is not intended for
real-world medical diagnosis.
