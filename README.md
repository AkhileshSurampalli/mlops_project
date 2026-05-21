# mlops_project
A robust pipeline for managing vehicle insurance data. This project deep dives into building and deploying a machine learning pipeline for real-world data management.

Project Setup and Structure:
1) Project Template: The template.py creates the initial project template, which includes the required folder structure and placeholder files.

Package Management:
1) Create setup.py and pyproject.toml files to import local packages.

Virtual Environment and Dependencies:
1) A virtual environment created to install dependencies from requirements.txt:
   conda create -n vehicle python=3.10 -y
   conda activate vehicle
   pip install -r requirements.txt

MongoDB Atlas Configuration:
1) Created a new project and set up a free M0 cluster and allowed the access from any IP address (0.0.0.0/0)
   
Pushing Data to Mongo DB:
1) Created a folder named notebook, added the dataset, and create a notebook file mongoDB_demo.ipynb
2) Used the notebook to push data to the MongoDB database.
3) Verify the data in MongoDB Atlas under Database > Browse Collections.

Logging, Exception Handling, and EDA
1) Created a test file called demo.py to create logging and exception handling modules.
2) Analyzed and engineered features in the EDA and Feature Engg notebook for further processing in the pipeline.

Data Ingestion Pipeline:
1) Define MongoDB connection funcitons in configuration.mongo_db_connections.py
2) Develop data ingestion components in the data_access and components.data_ingestion.py files to fetch and transform data.
3) Update entity/config_entity.py and entity/artifact_entity.py with relevant ingestion configurations.
4) Run demo.py after setting up MongoDB connection as an environment variable.
5) Set MongoDB URL:
   # For Bash
   export MONGODB_URL="mongodb+srv://<username>:<password>...."
   # For Powershell
   $env:MONGODB_URL = "mongodb+srv://<username>:<password>...."

Data validation, Transformation & Model Training
Data Validation:
1) Define schema in config.schema.yaml and implement data validation funcitons in utils.main_utils.py
Data transformation:
1) Implement data transformation logic in components.data_transformation.py and create estimator.py in the entity folder.
Model Training:
1) Define and implement model training steps in components.model_trainer.py using code from estimator.py



