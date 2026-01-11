# Polish-Equity-Analytics

## Project Description

This project analyzes the financial data of 370 Polish listed companies (GPW), taking into account sector-specific characteristics, to support investment decision-making. 

The goal is to build a robust classification model that accurately predicts each company's investment level (low, medium, or high) based on key financial indicators.

## How to Run

To run this project locally, you can use either Docker (recommended) or a standard Python environment.

### Using Docker 
* **Build the image:**
  ```bash
  docker build -t polish-equity-analytics .
* **Run the container:**
  ```bash
  docker run -p 8000:8000 polish-equity-analytics

### Using Python (Local Environment)
* **Install requirements:**
  ```bash
  pip install -r requirements.txt
* ** Launch FastAPI server:*
  ```bash
  uvicorn src.main:app --host 0.0.0.0 --port 8000

## Repository Structure
- `data/` - raw and labeled datasets (see `data/README.md`)
- `ETL/` - SQL scripts 
- `notebooks/` - Jupyter notebooks for preprocessing and modeling experiments
- `src/` - Python scripts for preprocessing data and training the final ML model
- `reports/` – Power BI dashboards and related analytics outputs
- `CRISP-DM.md` - detailed methodology workflow

## Tech Stack

- Language: Python 3.11
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, TensorFlow (MLP)
- Tools: Power BI, SQL (Data Extraction), Canva (Presentation)
- Backend & Deployment: FastAPI, Pydantic, Docker, Uvicorn, Joblib

For the full CRISP-DM methodology, see [CRISP-DM.md](CRISP-DM.md)
