# ❤️ Heart Failure Prediction using Machine Learning

This project applies multiple **machine learning classification models** to predict whether a patient has **heart disease** based on clinical and demographic features.
The goal is to compare different algorithms, evaluate them using appropriate metrics, and select the best-performing model for medical decision support.

---

## 📊 Dataset Overview

* **Total samples:** 918
* **Training set:** 734 samples
* **Test set:** 184 samples
* **Number of features:** 11
* **Target variable:** Presence of heart disease (binary classification)

The dataset contains patient information such as age, cholesterol, chest pain type, exercise-induced angina, and ECG-related measurements.

---

## ⚙️ Project Workflow

1. **Data Splitting**

   * Train–test split (80% / 20%)
   * Stratified sampling to preserve class balance
   * Test data kept completely unseen during training

2. **Feature Scaling**

   * Applied **StandardScaler**
   * Scaling performed **after** train–test split to avoid data leakage
   * Required for distance-based and margin-based models (KNN, SVM, Logistic Regression)

3. **Model Training**
   Six machine learning models were trained and evaluated:

   * Logistic Regression (baseline)
   * Decision Tree
   * Random Forest
   * Naive Bayes
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)

4. **Model Evaluation**
   Performance was evaluated using:

   * Confusion Matrix
   * Accuracy
   * Precision
   * Recall (Sensitivity)
   * **F1-Score** (primary metric due to medical importance of false negatives)

5. **Cross-Validation**

   * K-Fold Cross-Validation for robust performance estimation
   * Reduced variance caused by a single train–test split

6. **Hyperparameter Tuning**

   * GridSearchCV applied to the **top 3 models**
   * Tuned models:

     * Random Forest
     * SVM
     * KNN
   * Selection based on cross-validated F1-score

---

## 🏆 Results

### Best Performing Model

* **Model:** K-Nearest Neighbors (KNN)
* **Test F1-Score:** **0.9366**

### Top 3 Important Features

1. ST_Slope
2. ChestPainType
3. ExerciseAngina

These features align well with medical knowledge related to heart disease risk factors.

---

## 📈 Key Insights

* Accuracy alone is not sufficient for medical prediction problems.
* **F1-score** provides a better balance between precision and recall.
* Cross-validation and hyperparameter tuning significantly improve model reliability.
* Different models with similar CV scores can behave very differently on unseen test data.

---

## 🩺 Practical Applications

* Assist doctors in identifying patients at risk of heart disease
* Support early diagnosis and preventive care
* Potential foundation for a clinical decision-support system
* Can be extended into a patient-facing risk prediction tool

---

## 🔮 Future Improvements

* Handle missing or null values more robustly
* Add personalized lifestyle and dietary recommendations
* Deploy as a web application for real-time predictions
* Include explainability tools (e.g. SHAP) for better medical interpretability

---

## 📌 Final Remarks

This project demonstrates how machine learning can support healthcare decision-making when models are evaluated carefully and ethically.
While not a replacement for medical professionals, it can be a valuable **decision-support tool** with further validation and improvements.

---

## 🚀 Technologies Used

* Python
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Google Colab

---
