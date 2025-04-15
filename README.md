
 # Credit Default Prediction using Explainable AI (XAI)


This project builds a machine learning system to predict whether a person is likely to default on a loan. It demonstrates how real-world financial features can be used to develop accurate and explainable ML models.

---

## Intuitive Overview

- This was one of my first full machine learning builds.
- I was curious about how ML could improve credit default prediction—both in terms of predictive performance and transparency.
- I used the paper *“Enabling Machine Learning Algorithms for Credit Scoring – Explainable Artificial Intelligence (XAI) methods”* as inspiration for the structure of this project.
- The paper compares classical models (e.g., logistic regression) with modern ML models (e.g., Random Forest, XGBoost), and highlights the importance of explainability in financial decision-making using SHAP.
- I sourced a real dataset independently and built a similar pipeline:
  - Applied WOE transformations, log scaling, and imputation  
  - Trained and compared multiple models  
  - Used SHAP to explain individual and global predictions

---

## Algorithms Used – Purpose & Underlying Math

| Algorithm               | Purpose / Problem Solved                                         | Math at Lowest Abstraction Level                        |
|------------------------|------------------------------------------------------------------|---------------------------------------------------------|
| Logistic Regression     | Baseline classifier for binary outcomes                         | ŷ = 1 / (1 + e^(−Xβ))                                   |
| Random Forest           | Nonlinear ensemble classifier, reduces overfitting               | ŷ = majority_vote(Tree₁(X), Tree₂(X), ..., Treeₙ(X))   |
| XGBoost                 | Boosted trees for improving weak learners                        | ŷₜ = ŷₜ₋₁ + η × Tree(error)                             |
| Support Vector Machine  | Max-margin classifier for class separation                       | Maximize margin: min(½‖w‖²) subject to yᵢ(w·xᵢ + b) ≥ 1 |
| Naive Bayes             | Probabilistic model assuming feature independence                | P(y|X) ∝ P(X|y) × P(y)                                  |
| Linear Discriminant Analysis (LDA) | Projects data to maximize between-class separation     | Linear combination: wᵗx, with class covariance ratio    |

---

## Step-by-Step Process

1. **Data Cleaning**
   - Imputed missing values using median (`MonthlyIncome`, `NumberOfDependents`)
   - Removed extreme outliers in features like `MonthlyIncome`, `NumberOfTimes90DaysLate`, and others

2. **Feature Engineering**
   - Applied `log1p()` to skewed numerical features
   - Encoded categorical-like variables using Weight of Evidence (WOE)

3. **Scaling**
   - Used `StandardScaler` for models that are sensitive to feature magnitudes (SVM, Naive Bayes)

4. **Model Training**
   - Trained the following models:
     - Logistic Regression  
     - Random Forest (Default and Tuned)  
     - XGBoost  
     - Naive Bayes  
     - SVM  
     - LDA  
   - Used a stratified 80/20 train-test split
   - Evaluated models using AUC-ROC, F1-score, precision, and recall

5. **Explainability**
   - Used SHAP to interpret the best-performing model
   - Generated SHAP summary, dependence, and interaction plots

---

## Results

| Model                   | AUC-ROC | F1-Score | Accuracy | Notes                                 |
|------------------------|---------|----------|----------|----------------------------------------|
| Random Forest (Default)| 0.8597  | ~0.78    | 0.78     | Same core performance as tuned model   |
| Random Forest (Tuned)  | 0.8597  | ~0.78    | 0.78     | Best overall performance               |
| XGBoost                | 0.8540  | ~0.76    | 0.77     | Strong and consistent                  |
| LDA                    | 0.8463  | ~0.76    | 0.76     | Performs well under linear assumptions |
| Naive Bayes            | 0.8496  | ~0.75    | 0.75     | Lightweight and interpretable          |
| SVM                    | 0.8426  | ~0.74    | 0.78     | Sensitive to scaling                   |
| Logistic Regression    | 0.7720  | ~0.69    | 0.69     | Simple, interpretable baseline         |

- **Best Model**: Tuned Random Forest (highest AUC and balanced F1)

---

## Understanding SHAP Values

SHAP explains a model’s predictions by fairly attributing importance to each feature—based on how much it pushes the prediction up or down.

How it works:
- Tries all possible combinations of features (with and without the feature in question)
- Measures how much a feature changes the prediction
- Averages its effect across all combinations to compute its "credit"

Mathematically, the SHAP value for feature *i* is computed as:

φᵢ = ∑ [ (|S|! × (|F| − |S| − 1)!) / |F|! ] × [ f(S ∪ {i}) − f(S) ]  
for all subsets S ⊆ F \ {i}

Where:  
- F = full set of features  
- S = any subset excluding i  
- f(S) = model prediction using only the features in S

In practice:
- **Positive SHAP** → feature increases default risk  
- **Negative SHAP** → feature reduces default risk  
- **Near zero** → little to no influence

---

## Evaluation Metrics Summary

| Metric     | Value    | Meaning                                      |
|------------|----------|----------------------------------------------|
| AUC-ROC    | 0.8597   | Strong separation between classes            |
| F1-Score   | ~0.78    | Good balance between precision and recall    |
| Precision  | 0.79     | Few false positives                          |
| Recall     | 0.76     | Catches most defaults                        |
| Accuracy   | 0.78     | Overall correct predictions                  |

---

## Notes

- Visualizations and exploratory data analysis are provided in a separate notebook.
