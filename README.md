# Heart-Attack-prediction-using-ML
### Heart Attack Risk Prediction Project

This project aims to predict heart attack risk using a dataset of 303 patient records with 14 clinical features, including age, sex, chest pain type (cp), resting blood pressure (trtbps), cholesterol (chol), and maximum heart rate (thalachh). The target variable, `output`, indicates heart attack likelihood (0 = less chance, 1 = more chance). Implemented in a Jupyter notebook using Python, the project leverages Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn for data analysis and modeling.

**Exploratory Data Analysis (EDA)**: The dataset is inspected for quality, revealing no missing values and one duplicate row. Descriptive statistics highlight potential outliers in `chol` (max=564) and `oldpeak` (max=6.2). A count plot visualizes heart attack risk across age categories (`Age_CAT`: Adult, Middle_Age_Adult, Senior_Adult), showing risk patterns.

**Feature Engineering**: The `age` column is binned into `Age_CAT` (≤40, 41–60, >60) to capture age-related risk trends, enhancing model interpretability.

**Preprocessing**: Categorical features (e.g., cp, restecg, slp, caa, thall, Age_CAT) are one-hot encoded to convert them into numerical format, with the first category dropped to avoid multicollinearity. Numerical features (age, trtbps, chol, thalachh, oldpeak) are standardized using `StandardScaler` to ensure uniform scales, mitigating outlier impact.

**Modeling**: The dataset is split into 80% training (242 samples) and 20% testing (61 samples) sets. Three models—Logistic Regression, Decision Tree, and Random Forest—are trained and evaluated using 10-fold cross-validation. Metrics include accuracy, AUC, precision, recall, and F1-score. Logistic Regression performs best (Accuracy: 0.858, AUC: 0.9218, F1: 0.8721), followed by Random Forest (Accuracy: 0.7855) and Decision Tree (Accuracy: 0.7396).

**Conclusion**: Logistic Regression is the most effective model for heart attack risk prediction. Future improvements include outlier handling, feature selection, and hyperparameter tuning to enhance performance in this clinically relevant machine learning pipeline.

*Word count: 300*
