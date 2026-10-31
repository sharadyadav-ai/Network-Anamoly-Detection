# 🛡️ Network Anomaly Detection System Using Machine Learning

An intelligent Network Anomaly Detection System that analyzes network traffic and identifies suspicious activities using Machine Learning algorithms.

The project uses the **UNSW-NB15 dataset** and implements two machine learning algorithms: Random Forest and Isolation Forest to detect malicious and abnormal network behavior.

## 📌 Project Overview

Traditional network security systems often rely on predefined rules and signatures to detect cyberattacks. These systems may fail to identify new or unusual network activities.

This project uses machine learning to analyze network traffic, learn normal communication patterns, and identify potentially suspicious behavior.

The system provides a dashboard to visualize network traffic, detect anomalies, and analyze model predictions.

## 🎯 Objectives

- Analyze network traffic using machine learning.
- Identify normal and abnormal network activities.
- Detect potentially malicious network behavior.
- Compare supervised and unsupervised learning algorithms.
- Visualize detection results through an interactive dashboard.
- Generate reports for detected anomalies.

## ✨ Features

- Network traffic dataset analysis.
- Data preprocessing and feature extraction.
- Machine learning-based anomaly detection.
- Random Forest classification.
- Isolation Forest anomaly detection.
- Model performance evaluation.
- Interactive Streamlit dashboard.
- Traffic prediction and visualization.
- Downloadable detection reports.

## 🏗️ System Architecture

```text
       Network Traffic Dataset
                 |
                 v
         Data Preprocessing
                 |
                 v
         Feature Extraction
                 |
                 v
          Machine Learning
                 |
          +------+------+
          |             |
          v             v
     Random Forest  Isolation Forest
          |             |
          v             v
     Attack/Normal   Anomaly/Normal
          |             |
          +------+------+
                 |
                 v
         Streamlit Dashboard
                 |
                 v
        Detection Reports
```

## 🧠 Machine Learning Algorithms

### 1. Random Forest

Random Forest is a supervised machine learning algorithm used to classify network traffic into:

- Normal Traffic
- Attack Traffic

It learns from labeled network traffic records.

### 2. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm.

It learns patterns from normal network traffic and identifies unusual observations that differ from those patterns.

## 📊 Dataset

**Dataset:** UNSW-NB15

The UNSW-NB15 dataset contains normal and malicious network traffic records.

It includes different attack categories, such as:

- DoS
- Exploits
- Fuzzers
- Reconnaissance
- Generic
- Analysis
- Backdoors
- Shellcode
- Worms

Dataset source:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

The training CSV used in this project contains 45 columns, including network traffic features and target labels.

**Target column:**
- `0` – Normal traffic
- `1` – Attack traffic

Note: The dataset is not included in this repository. Download it separately and place the required CSV file in the project directory.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Scapy | Network packet analysis (future integration) |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |
| SQLite | Future alert storage |

## 📁 Project Structure

```text
Network-Anomaly-Detection/
│
├── dataset/
│
├── models/
│   ├── random_forest_pipeline.joblib
│   ├── isolation_forest_pipeline.joblib
│   └── feature_columns.json
│
├── reports/
│   └── evaluation.json
│
├── train_model.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Network-Anomaly-Detection.git
```

### Step 2: Navigate to the project directory

```bash
cd Network-Anomaly-Detection
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Install dependencies

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, use:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 5: Add the dataset

Download the UNSW-NB15 training dataset and place the CSV file in the project directory.

Expected filename:

```text
UNSW_NB15_training-set(in).csv
```

### Step 6: Train the machine learning models

```bash
python train_model.py --data "UNSW_NB15_training-set(in).csv"
```

The training script will:

- Load the dataset.
- Preprocess numerical and categorical features.
- Split the dataset into training and testing sets.
- Train the Random Forest model.
- Train the Isolation Forest model using normal training records.
- Evaluate both models.
- Save the trained models.

### Step 7: Run the dashboard

```bash
streamlit run dashboard.py
```

Open the local Streamlit URL displayed in your terminal.

Upload a compatible network traffic CSV to analyze its predictions.

## 📈 Model Evaluation

The models are evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation results are saved in:

```text
reports/evaluation.json
```

Actual model performance will depend on the dataset, preprocessing, and model configuration.

## 🚀 Future Improvements

The project can be extended with:

- Real-time network traffic monitoring.
- Live packet capture using Scapy.
- Automatic security alerts.
- IP address-based traffic analysis.
- Attack severity classification.
- Historical anomaly monitoring.
- SQLite-based alert management.
- Advanced deep learning models such as Autoencoders.
- Cross-dataset model evaluation.

## 🔐 Security and Ethical Considerations

- Monitor only networks for which you have authorization.
- Use isolated lab environments for security testing.
- Treat anomaly detections as suspicious indicators, not proof of an attack.
- Protect collected network information and logs.

## 👨‍💻 Project Information

**Project Name:** Network Anomaly Detection System Using Machine Learning

**Domain:** Network Security and Machine Learning

**Programming Language:** Python

**Dataset:** UNSW-NB15

**Algorithms:** Random Forest and Isolation Forest

**Interface:** Streamlit

## 📜 License

This project is intended for educational and research purposes.

You can add an MIT License if you want to allow others to reuse and modify your code under its terms.

---

⭐ If you find this project useful, consider giving the repository a star!
