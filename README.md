# 🚗 MLOps Vehicle Insurance Prediction Pipeline

A production-ready end-to-end MLOps pipeline for managing, training, evaluating, and deploying machine learning models using vehicle insurance data.

This project demonstrates how to build a scalable machine learning system with complete lifecycle management including:

* Data ingestion from MongoDB Atlas
* Data validation and transformation
* Model training and evaluation
* AWS S3 model storage
* Docker containerization
* CI/CD with GitHub Actions
* Deployment on AWS EC2

---

# 📌 Project Overview

The goal of this project is to implement a robust and modular MLOps architecture capable of handling real-world machine learning workflows.

The pipeline automates:

1. Data ingestion from MongoDB
2. Validation and preprocessing
3. Feature engineering
4. Model training and evaluation
5. Model versioning and storage in AWS S3
6. API deployment using Flask
7. CI/CD automation with GitHub Actions
8. Docker-based deployment on AWS EC2

---

# 🏗️ Project Architecture

```text
MongoDB Atlas
      │
      ▼
Data Ingestion Pipeline
      │
      ▼
Data Validation
      │
      ▼
Data Transformation
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
AWS S3 Model Registry
      │
      ▼
Prediction Pipeline
      │
      ▼
Flask API
      │
      ▼
Docker Container
      │
      ▼
AWS EC2 Deployment
```

---

# 📂 Project Structure

```text
MLOps_Project/
│
├── notebook/
│   ├── data/
│   ├── mongoDB_demo.ipynb
│   └── EDA_Feature_Engineering.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   │
│   ├── configuration/
│   │   └── mongo_db_connections.py
│   │
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   │
│   ├── aws_storage/
│   ├── utils/
│   ├── logger/
│   └── exception/
│
├── static/
├── templates/
├── app.py
├── demo.py
├── requirements.txt
├── setup.py
├── pyproject.toml
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# ⚙️ Tech Stack

## Programming & ML

* Python 3.10
* Scikit-learn
* Pandas
* NumPy

## Database

* MongoDB Atlas

## Cloud & Deployment

* AWS EC2
* AWS S3
* AWS ECR

## DevOps & CI/CD

* Docker
* GitHub Actions

## Web Framework

* Flask

---

# 🚀 Installation

## Clone the Repository

```bash
git clone https://github.com/yourusername/MLOps_Project.git
cd MLOps_Project
```

---

# 🐍 Create Virtual Environment

## Using Conda

```bash
conda create -n vehicle python=3.10 -y
conda activate vehicle
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🌐 MongoDB Atlas Setup

## Step 1: Create MongoDB Atlas Cluster

* Create a free M0 cluster
* Allow access from:

```text
0.0.0.0/0
```

* Create database credentials

---

## Step 2: Set MongoDB Environment Variable

### Bash

```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/"
```

### PowerShell

```powershell
$env:MONGODB_URL = "mongodb+srv://<username>:<password>@cluster.mongodb.net/"
```

---

# 📊 Push Dataset to MongoDB

1. Add dataset to:

```text
notebook/data/
```

2. Run:

```text
mongoDB_demo.ipynb
```

3. Verify data in MongoDB Atlas:

```text
Database → Browse Collections
```

---

# 🔍 Exploratory Data Analysis & Feature Engineering

EDA and feature engineering were performed to:

* Analyze feature distributions
* Handle missing values
* Engineer relevant predictive features
* Prepare data for transformation and training

Notebook used:

```text
EDA_Feature_Engineering.ipynb
```

---

# 🧩 Pipeline Components

## 1. Data Ingestion

Responsible for:

* Connecting to MongoDB Atlas
* Fetching data
* Converting data into usable format

Files:

* `mongo_db_connections.py`
* `data_ingestion.py`


## 2. Data Validation

Responsible for:

* Schema validation
* Data consistency checks
* Missing value verification

Files:

* `schema.yaml`
* `main_utils.py`


## 3. Data Transformation

Responsible for:

* Feature preprocessing
* Encoding
* Scaling
* Transformation pipelines

Files:

* `data_transformation.py`
* `estimator.py`

---

## 4. Model Training

Responsible for:

* Training ML models
* Hyperparameter handling
* Saving trained artifacts

File:

* `model_trainer.py`

---

## 5. Model Evaluation

Responsible for:

* Comparing model performance
* Accepting/rejecting models
* Tracking best models

---

## 6. Model Deployment

Responsible for:

* Pushing models to AWS S3
* Pulling latest production models
* Serving predictions through Flask API

Files:

* `model_pusher.py`
* `s3_estimator.py`
* `app.py`

---

# ☁️ AWS Setup

## IAM Configuration

Create an IAM user and configure:

* AWS Access Key
* AWS Secret Key

---

# 🪣 AWS S3 Bucket

Create an S3 bucket:

```text
my-model-mlopsproj
```

Region:

```text
us-east-1
```

---

# 🔐 Set AWS Credentials

Configure environment variables:

```bash
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"
```

---

# 🐳 Docker Setup

## Build Docker Image

```bash
docker build -t vehicle-mlops .
```

## Run Container

```bash
docker run -p 5000:5000 vehicle-mlops
```

---

# 🔄 CI/CD with GitHub Actions

GitHub Actions automates:

* Build process
* Docker image creation
* Deployment pipeline

## Required GitHub Secrets

Add the following secrets:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_DEFAULT_REGION`
* `ECR_REPO`

---

# 🖥️ AWS EC2 Deployment

## EC2 Setup

* Launch EC2 instance
* Install Docker
* Configure security groups
* Open port:

```text
5000
```

---

# 🔗 Connect EC2 as GitHub Self-Hosted Runner

Configure GitHub Actions runner on EC2 for automated deployment.

---

# 🌍 Access Deployed Application

```text
http://<public_ip>:5000
```

---

# 🧪 Run the Application Locally

```bash
python app.py
```

---

# 📈 Future Improvements

* Add MLflow experiment tracking
* Implement Kubernetes deployment
* Add model monitoring and drift detection
* Integrate FastAPI
* Add unit and integration testing
* Implement automated retraining pipeline

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a pull request

---

# 📝 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed as part of an end-to-end MLOps learning and deployment project focused on real-world machine learning infrastructure and automation.
