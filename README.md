# 🌾 Crop Recommendation System

An AI/ML project that recommends the best crop to grow based on soil nutrients
and climate conditions using **Random Forest** and **KNN** classifiers.

---

## 📁 Project Structure

```
crop_project/
├── app.py                      ← Streamlit web app
├── train_model.py              ← Model training script
├── crop_notebook.ipynb         ← Jupyter notebook (EDA + preprocessing + evaluation)
├── Crop_recommendation.csv     ← Dataset (2300 samples, 23 crops)
├── requirements.txt
├── models/
│   ├── rf_model.pkl            ← Trained Random Forest
│   ├── knn_model.pkl           ← Trained KNN
│   ├── scaler.pkl              ← StandardScaler
│   └── label_encoder.pkl       ← LabelEncoder
├── plots/
│   ├── feature_dist.png
│   ├── correlation.png
│   ├── crop_counts.png
│   ├── confusion_matrix_rf.png
│   ├── confusion_matrix_knn.png
│   ├── feature_importance.png
│   └── model_comparison.png
└── classification_report.txt
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_model.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### 4. (Optional) Open the notebook
```bash
jupyter notebook crop_notebook.ipynb
```

---

## 🌱 Dataset

| Feature       | Description                          | Unit    |
|---------------|--------------------------------------|---------|
| N             | Nitrogen ratio in soil               | mg/kg   |
| P             | Phosphorus ratio in soil             | mg/kg   |
| K             | Potassium ratio in soil              | mg/kg   |
| temperature   | Average temperature                  | °C      |
| humidity      | Relative humidity                    | %       |
| ph            | Soil pH value                        | —       |
| rainfall      | Annual rainfall                      | mm      |
| label         | Crop name (target)                   | —       |

- **2300 samples**, **23 crops**, **0 missing values**
- Source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## 🤖 Models & Results

| Model         | Test Accuracy | 5-Fold CV       |
|---------------|---------------|-----------------|
| Random Forest | **90.65%**    | ~89.8% ± 1.2%   |
| KNN           | 84.57%        | ~83.4% ± 1.8%   |

**Random Forest** is selected as the primary model.

---

## 🛠️ Technologies

- Python 3.10+
- scikit-learn — ML models
- pandas / NumPy — data processing
- matplotlib / seaborn — visualizations
- Streamlit — web interface
- joblib — model serialization

---

## 👨‍🎓 Project Requirements Met

- [x] AI/ML clearly used (Random Forest + KNN)
- [x] Dataset: 2300+ labeled samples
- [x] Data preprocessing (StandardScaler, LabelEncoder, train/test split)
- [x] At least 1 ML algorithm (2 implemented)
- [x] Model evaluation: accuracy, confusion matrix, classification report
- [x] Graphs: feature distributions, correlation, feature importance, model comparison
- [x] User interface: Streamlit web app with 3 tabs
- [x] Prediction output with confidence score and alternative crops
