import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_data(num_patients=500, max_visits_per_patient=5):
    """Generates synthetic patient visit records with realistic comorbidities."""
    records = []
    
    # Define comorbidities and treatments
    base_conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Kidney Disease', 'Allergy']
    
    # Biases/Rules
    # Diabetes <-> Hypertension
    # Asthma -> Inhaler + Steroid
    # Heart Disease -> Statin + Beta Blocker
    
    patient_ids = [f"P{str(i).zfill(4)}" for i in range(1, num_patients + 1)]
    
    visit_counter = 1
    
    for pid in patient_ids:
        # Assign an age group
        age_val = random.randint(18, 90)
        if age_val < 35:
            age_group = "Young"
        elif age_val < 60:
            age_group = "Adult"
        else:
            age_group = "Senior"
            
        num_visits = random.randint(1, max_visits_per_patient)
        
        # Patient base probability for conditions
        has_diabetes = random.random() < (0.3 if age_group == "Senior" else 0.1)
        has_hypertension = random.random() < 0.8 if has_diabetes else random.random() < 0.2
        has_asthma = random.random() < 0.15
        has_heart_disease = random.random() < (0.25 if age_group == "Senior" else 0.05)
        
        for v in range(num_visits):
            vid = f"V{str(visit_counter).zfill(5)}"
            visit_counter += 1
            
            # Events for this visit
            events = []
            
            if has_diabetes:
                events.append("Diabetes")
                # sometimes get metformin
                if random.random() < 0.7:
                    events.append("Metformin")
            
            if has_hypertension:
                events.append("Hypertension")
                if random.random() < 0.6:
                    events.append("Lisinopril")
            
            if has_asthma:
                events.append("Asthma")
                events.append("Inhaler")
                if random.random() < 0.4:
                    events.append("Steroid")
                    
            if has_heart_disease:
                events.append("Heart Disease")
                events.append("Statin")
                if random.random() < 0.8:
                    events.append("Beta Blocker")
            
            # Add random noise events
            if random.random() < 0.2:
                events.append("Allergy")
            if random.random() < 0.05:
                events.append("Kidney Disease")
                
            # If no event, just routine checkup
            if not events:
                events.append("Routine Checkup")
                
            for event in events:
                records.append({
                    "PatientID": pid,
                    "VisitID": vid,
                    "AgeGroup": age_group,
                    "MedicalEvent": event
                })

    df = pd.DataFrame(records)
    return df

def main():
    print("Generating raw data...")
    df_raw = generate_synthetic_data(num_patients=1000, max_visits_per_patient=4)
    
    # Save raw data
    raw_path = "workspace/data_raw/raw_data.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path} ({len(df_raw)} records)")
    
    # Preprocessing
    print("Preprocessing data...")
    df_clean = df_raw.copy()
    
    # Remove duplicates and nulls
    df_clean.drop_duplicates(inplace=True)
    df_clean.dropna(inplace=True)
    
    # Normalize text (strip spaces, capitalize consistently)
    df_clean["MedicalEvent"] = df_clean["MedicalEvent"].str.strip().str.title()
    
    # Feature Engineering
    # Calculate Visit Frequency per patient
    visit_counts = df_clean.groupby("PatientID")["VisitID"].nunique().reset_index()
    visit_counts.rename(columns={"VisitID": "VisitFrequency"}, inplace=True)
    df_clean = df_clean.merge(visit_counts, on="PatientID", how="left")
    
    # Save processed data
    processed_path = "workspace/data_processed/processed_data.csv"
    df_clean.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
    
    # Transaction Conversion
    print("Converting to transaction format...")
    # Group by VisitID
    
    # We want features in the transaction. E.g. {Senior, Diabetes, Hypertension, Statin}
    # AgeGroup is a feature, we can just append it as an item.
    df_clean["AgeGroupItem"] = "Age_" + df_clean["AgeGroup"]
    
    # Aggregate items per visit
    # We will combine MedicalEvents and AgeGroup (unique per visit) into a list
    transactions = []
    grouped = df_clean.groupby("VisitID")
    
    for vid, group in grouped:
        items = set(group["MedicalEvent"].tolist())
        # Add the age group
        age_group = group["AgeGroupItem"].iloc[0]
        items.add(age_group)
        
        # Add Visit Frequency categorization
        freq = group["VisitFrequency"].iloc[0]
        if freq == 1:
            items.add("Freq_Single")
        elif freq <= 3:
            items.add("Freq_Low")
        else:
            items.add("Freq_High")
            
        transactions.append({
            "VisitID": vid,
            "Items": ",".join(sorted(list(items)))
        })
        
    df_trans = pd.DataFrame(transactions)
    trans_path = "workspace/transactions/transactions.csv"
    df_trans.to_csv(trans_path, index=False)
    print(f"Transactions saved to {trans_path} ({len(df_trans)} transactions)")

if __name__ == "__main__":
    main()
