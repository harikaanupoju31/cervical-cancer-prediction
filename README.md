# 🩺 Cervical Cancer Early-Risk Prediction System

An AI/ML-based project for estimating cervical cancer risk from selected patient and clinical information.

> **Note:** This is an educational/project demonstration. The prediction is a machine-learning model score and is **not a medical diagnosis**.

## 📌 Project Overview

The **Cervical Cancer Early-Risk Prediction System** uses machine learning to estimate a patient's cervical cancer risk from input factors such as:

- Age
- Number of sexual partners
- Age at first sexual intercourse
- Number of pregnancies
- Smoking history
- Smoking duration and packs/year
- Hormonal contraceptive history
- IUD history
- STD history
- STD-related information
- HPV-related information
- Previous diagnosis indicators

The project provides a Streamlit web interface where the user enters patient information and receives a model-estimated risk score.

## ✨ Features

- 🩺 Patient information input
- 🦠 STD-related information
- 🔬 Previous diagnosis indicators
- 🤖 Machine-learning-based prediction
- 📊 Model-estimated risk percentage
- 🟢 Lower / 🟠 Moderate / 🔴 Higher model-estimated risk category
- 🌿 Personalized educational guidance
- 📋 Patient factor summary
- 🔧 View model input features
- 💻 Streamlit web application

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Machine Learning

## 📁 Project Structure

```text
cervical-cancer-prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── cervical-cancer.csv
