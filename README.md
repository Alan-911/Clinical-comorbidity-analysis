# Clinical Comorbidity Analysis Project

**Discovery of Comorbidity and Treatment Patterns in Clinical Records**

This project performs Association Rule Mining (ARM) on synthetic healthcare data to identify comorbidity and treatment patterns.

## Project Structure
- `scripts/`: Python scripts for data generation, association rule mining, and visualizations.
- `workspace/`: Simulates the external hard drive attached to the lab server.
  - `data_raw/`: Raw generated patient visit data.
  - `data_processed/`: Cleaned and feature-engineered datasets.
  - `transactions/`: Transaction-formatted dataset grouped by VisitID.
  - `dolt_repo/`: Dolt repository for dataset versioning (raw, cleaned, engineered datasets).
  - `outputs/`: Frequent itemsets, association rules, and Apriori vs FP-Growth comparison reports.
  - `visualizations/`: Generated comorbidity network graphs.
  - `streamlit_app/`: High-end interactive Streamlit dashboard.

## Setup Instructions
1. Install Python 3.10+ and the required packages:
   ```bash
   pip install -r workspace/streamlit_app/requirements.txt
   ```
2. Download Dolt from [DoltHub](https://www.dolthub.com/download) and add it to your PATH if you wish to inspect the dataset versions.

## Execution Workflow
To re-run the entire pipeline from scratch, execute the following scripts in order:

1. **Generate and Process Data:**
   ```bash
   python scripts/1_generate_and_process_data.py
   ```
2. **Run Association Rule Mining (Apriori & FP-Growth):**
   ```bash
   python scripts/2_run_association_mining.py
   ```
3. **Generate Network Visualization:**
   ```bash
   python scripts/3_generate_visualizations.py
   ```

## Running the Dashboard
To start the interactive presentation dashboard locally:
```bash
cd workspace/streamlit_app
streamlit run app.py
```

## Deployment
To deploy this project for free on Streamlit Cloud:
1. Ensure all code is pushed to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select this repository and choose `workspace/streamlit_app/app.py` as the Main file path.
5. Click **Deploy**. Your dashboard will be live and accessible publicly.
