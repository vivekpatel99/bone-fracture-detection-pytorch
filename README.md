# Lung and Colon Cancer Classification

## About Dataset

This dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.
The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.
There are five classes in the dataset, each with 5,000 images, being:

- Lung benign tissue
- Lung adenocarcinoma
- Lung squamous cell carcinoma
- Colon adenocarcinoma
- Colon benign tissue

How to Cite this Dataset
If you use in your research, please credit the author of the dataset:

Original Article
Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019

Relevant Links
https://arxiv.org/abs/1912.12142v1
https://github.com/tampapath/lung_colon_image_set
Dataset BibTeX
@article{,
title= {LC25000 Lung and colon histopathological image dataset},
keywords= {cancer,histopathology},
author= {Andrew A. Borkowski, Marilyn M. Bui, L. Brannon Thomas, Catherine P. Wilson, Lauren A. DeLand, Stephen M. Mastorides},
url= {https://github.com/tampapath/lung_colon_image_set}
}

## 🛠️ Technologies Used

- **Python:** Core programming language.
- **UV:** Environment management.
- **Jupyter Notebook:** For exploratory data analysis (EDA), model development, and training.
- **AWS:** `IAM`, `S3`, `ECR (Amazon Elastic Container Registry)`, `EC2` for cloud infrastructure, model storage, and deployment.
- **Streamlit:** For building the prediction API (`app.py`).
- **Docker:** Containerization for consistent environment.
- **GitHub Actions:** CI/CD automation.
- **Pytorch/Pytorch lightning:** data preparation & modeltraining the model

## Project Structure

```tree

```

### Model Training Pipeline

### CI/CD Pipeline

## 🔧 Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vivekpatel99/lung-and-colon-cancer-classification-pytorch.git
   cd heart-attack-predictor-end-to-end-ml-project
   ```

   *(Note: The project structure is assumed to be pre-generated, potentially using a `template.py` script as mentioned in development steps.)*

2. **Create and Activate Virtual Environment (using `uv`)/Setup Project Environment:**

   *First, ensure you have `uv` installed. If not, follow the instructions here.*

   ```bash
   # uv will create a virtual environment named .venv and install dependencies
   uv sync
   ```

3. **Set up MongoDB Atlas:**

   - Sign up/log in to MongoDB Atlas.
   - Create a new project.
   - Create a free tier (M0) cluster.
   - Configure a database user (username/password).
   - Go to "Network Access" and add `0.0.0.0/0` to allow connections from any IP (for development; restrict in production).
   - Go to your cluster > "Connect" > "Drivers" > Select Python (3.6 or later).
   - Copy the connection string.

4. **Set up AWS:**

   - Log in to your AWS Console.
   - **IAM User:**
     - Go to IAM > Users > Create user (e.g., `heartattack-dev-user`).
     - Attach policies: `AdministratorAccess` (for ease of setup, consider more restrictive policies for production).
     - Create access key (select "Command Line Interface (CLI)") and download/copy the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
   - **S3 Bucket:**
     - Go to S3 > Create bucket.
     - Choose a unique bucket name (e.g., `yourname-heartattack-models`).
     - Select the region (e.g., `us-east-1`).
     - Uncheck "Block all public access" and acknowledge (needed for simple deployment setup; review security implications).
     - Create the bucket.

5. **Set up Environment Variables:**

   - **Method 1: `.env` file (Recommended for Local Development)**

     - Create a `.env` file in the project root.
     - Add your credentials and configurations:

     ```dotenv
     # .env
     AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
     AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
     AWS_REGION="us-east-1" # Or your chosen region
     # Add other necessary variables like DB_NAME, COLLECTION_NAME, BUCKET_NAME if needed by the code
     # BUCKET_NAME="yourname-heartattack-models"
     ```

     - *Note: Ensure `.env` is listed in your `.gitignore` file. If you are using `uv run`, `uv` will automatically load variables from a `.env` file in the current or parent directories, so manual exporting might not be necessary when running scripts via `uv run`.*
     - The application code needs to be configured to load variables from `.env` (e.g., using `python-dotenv`).

   - **Method 2: Exporting Variables (Common for Terminals/CI/CD)**

     - **Bash/Zsh:**

     ```bash
     export MONGODB_URL="mongodb+srv://<username>:<password>@..."
     export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
     export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
     export AWS_REGION="us-east-1"
     # Check: echo $MONGODB_URL
     ```

     - **PowerShell:**

     ```powershell
     $env:MONGODB_URL = "mongodb+srv://<username>:<password>@..."
     $env:AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_ID"
     $env:AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_ACCESS_KEY"
     $env:AWS_REGION = "us-east-1"
     # Check: echo $env:MONGODB_URL
     ```

     - **Windows CMD (or System Environment Variables GUI):**

     ```cmd
     set MONGODB_URL="mongodb+srv://<username>:<password>@..."
     set AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
     # ... etc
     ```

6. **Run the Project**

   ```bash
   uv run streamlit run app.py
   ```

7. **Push new changes to AWS**

   - First uncomment code from [Github Actions](.github/workflows/aws.yaml) for AWS.
   - the commit and push your changes. Github Action will automatically build and deploy project on `EC2` instance.

## 💾 Data Handling

- **Dataset Loading:** The initial dataset (from Kaggle) should be placed appropriately (e.g., in a `data/` or `notebooks/` directory).
- **MongoDB Upload:** Use the provided Jupyter notebook (e.g., `notebooks/mongoDB_demo.ipynb`) to upload the dataset to your configured MongoDB Atlas database.
- **Data Ingestion:** The pipeline's `DataIngestion` component reads data from MongoDB.

## 🏋️ Running the Training Pipeline

- Ensure your environment variables are set correctly (MongoDB URL, AWS Keys).
- Execute the main training pipeline script (e.g., `python main.py` or `python demo.py` - adjust command based on your project's entry point).
- This script orchestrates the following components:
  1. **Data Ingestion:** Fetches data from MongoDB.
  2. **Data Validation:** Checks data against a predefined schema (`config/schema.yaml`).
  3. **Data Transformation:** Performs preprocessing and feature engineering.
  4. **Model Training:** Trains the machine learning model.
  5. **Model Evaluation:** Evaluates the trained model against metrics and potentially a baseline model from S3.
  6. **Model Pusher:** If the model meets evaluation criteria, pushes the model artifacts (model file, preprocessor) to the configured AWS S3 bucket.

## 📈 Results & Visualizations

## 🖥️ Hardware Specifications

NOTE: To enhance ball detection accuracy, the YOLOv12L model was trained using an image size of 1280x1280. Training was performed using Lightning Studio to ensure sufficient GPU memory was available. The project was developed and tested on the following hardware:
it was developed and tested on the following hardware:

- **CPU:** AMD Ryzen 5900X
- **GPU:** NVIDIA GeForce RTX 3080 (10GB VRAM)
- **RAM:** 32 GB DDR4

While these specifications are recommended for optimal performance, the project can be adapted to run on systems with less powerful hardware.

## 📚 Reference

1. [hydra-example-project](https://github.com/FlorianWilhelm/hydra-example-project)
2. [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data)
