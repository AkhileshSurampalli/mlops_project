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



