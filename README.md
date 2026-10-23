<div align="center">

# 🎓 Smart Outcome Predictor

### Ensemble Machine Learning for Student Completion & Score Prediction

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Project-Machine%20Learning-6C63FF?style=for-the-badge" alt="Machine Learning">
</p>

**A practical ML dashboard that compares ensemble models to predict course completion and estimate final student scores.**

</div>

---

## 📌 Project Overview

**Smart Outcome Predictor** is a Streamlit-based machine learning application built to explore ensemble learning techniques on student-related data. It supports two prediction tasks:

- **Classification:** predicts whether a student will complete a course using `completion_status` (`0 = Not Completed`, `1 = Completed`).
- **Regression:** estimates a student's `final_score` on a 0–100 scale.

The dashboard lets users preview the dataset, train and compare models, inspect evaluation metrics, and try predictions using editable feature values.

> ⚠️ This project is intended for educational use. Predictions should not be treated as definitive judgments about a student or used for consequential decisions without careful validation, fairness checks, and human oversight.

## ✨ Key Features

- 📂 Upload a dataset in **CSV or Excel** format, or use the included Excel dataset.
- 🔎 Preview rows, columns, and missing-value counts.
- ⚙️ Choose the test-set percentage and random state.
- 🎯 Train models for both classification and regression.
- 📊 Compare model metrics in tables and charts.
- 🧑‍🎓 Try interactive course-completion and final-score predictions.
- 🧠 Learn the differences between **Bagging, Boosting, Voting, and Stacking**.
- ⬇️ Download classification and regression comparison tables as CSV files.

## 🧠 Machine Learning Techniques

| Technique / Model | Purpose |
|---|---|
| Logistic Regression | Classification baseline |
| Ridge Regression | Regression baseline |
| Random Forest | Tree-based ensemble |
| Bagging | Combines models trained on bootstrap samples |
| AdaBoost | Boosting approach that focuses on prior errors |
| Histogram Gradient Boosting | Gradient-boosted tree model |
| Voting | Combines predictions from multiple estimators |
| Stacking | Uses a meta-model to combine base-model predictions |
| XGBoost *(optional at runtime)* | Additional boosting model when installed and available |

The app selects the displayed best model using **F1-score** for classification and the **lowest RMSE** for regression, based on the held-out test split.

## 📏 Evaluation Metrics

**Classification**
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (when available)

**Regression**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

Lower MAE/RMSE and a higher R² score generally indicate stronger regression performance. Metrics depend on the dataset and split; results may vary.

## 🗂️ Project Structure

```text
pr_5/
├── main.py
├── model/
│   └── model.py
├── data/
│   └── Smart_Outcome_Predictor_Dataset_5200.xlsx
├── requirements.txt
├── .gitignore.txt
├── Smart_Outcome_Predictor_Theory.pdf
├── Readme.md
└── Smart_Outcome_Predictor_Meet_mehta_12237.mp4
```

> Keep the dataset in `data/` if you want the app to load it automatically. Alternatively, upload a CSV or Excel file from the app sidebar.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Meet290710/Smart-Outcome-Predictor.git
cd Smart-Outcome-Predictor
```

Replace the repository URL above if you use a different GitHub repository name.

### 2. Create and activate a virtual environment (recommended)

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit app

```bash
streamlit run main.py
```

The app should open in your browser. If it does not, use the local URL printed in the terminal.

## 🧭 How to Use

1. Launch the app.
2. Keep the included dataset or upload a `.csv`, `.xlsx`, or `.xls` file.
3. Confirm that the dataset contains `completion_status` and `final_score` columns.
4. Choose the test-data percentage and random state from the sidebar.
5. Select **Train & Evaluate Models**.
6. Review classification and regression metrics.
7. Enter feature values to try a prediction.
8. Download the metric comparison tables if needed.



## 🛠️ Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Visualization:** Streamlit charts, Matplotlib
- **Dataset support:** Excel and CSV via Pandas / OpenPyXL

## 📁 Dataset Requirements

The application expects these target columns:

| Column | Meaning |
|---|---|
| `completion_status` | Course completion label (`0` or `1`) |
| `final_score` | Final score target, expected in the range 0–100 |

Other columns can be used as input features. The code excludes `student_id` and `course_start_date` from model features and performs basic missing-value handling and categorical encoding.

**Privacy note:** Before publishing a dataset publicly, confirm that you have permission to share it and remove personally identifiable or sensitive student information.

## 🔮 Possible Future Improvements

- Add cross-validation and hyperparameter tuning.
- Include confusion matrices and ROC curves.
- Add explainability tools such as feature importance or SHAP.
- Improve fairness checks and class-imbalance handling.
- Add model persistence to avoid retraining on every fresh session.
- Deploy the app and add a live demo link.

## 👨‍💻 Author

**Meet Mehta**  
BCA Student | Machine Learning & Data Analytics

- GitHub: [@Meet290710](https://github.com/Meet290710)

---

<div align="center">

**Built as a hands-on learning project in Ensemble Machine Learning.** 🚀

</div>
