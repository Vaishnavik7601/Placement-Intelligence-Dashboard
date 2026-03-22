# Placement Intelligence Dashboard

A complete Data Analytics and Machine Learning project analyzing placement factors for BCA/MCA/CS students.

---
Live App: https://placement-intelligence-dashboard-zah6gzty5eenfaj6na7ebp.streamlit.app/


## Project Overview

The Placement Intelligence Dashboard is a Data Analytics and ML project that analyzes 10,000 student records to uncover key factors affecting campus placements. It combines Data Analytics, Power BI Visualization, and Machine Learning to predict whether a student will get placed based on their academic and personal profile.

---

## Key Highlights

| Metric | Value |
|---|---|
| Dataset Size | 10,000 students |
| ML Model Accuracy | 99.85% (Random Forest) |
| Total Visualizations | 14 charts |
| Power BI Pages | 7 pages |
| Placement Rate | 16.59% |

---

## Project Structure

```
PLACEMENTDASHBOARD/
|
|-- data/
|   |-- rawdata.csv              # Original Kaggle dataset
|   |-- cleaned_data.csv         # Cleaned & processed data
|
|-- visuals/
|   |-- 01_day1_overview.png
|   |-- 02_cgpa_vs_placement.png
|   |-- 03_iq_vs_placement.png
|   |-- 04_communication_vs_placement.png
|   |-- 05_projects_vs_placement.png
|   |-- 06_internship_impact.png
|   |-- 07_correlation_heatmap.png
|   |-- 08_cgpa_band_placement_rate.png
|   |-- 09_iq_band_placement_rate.png
|   |-- 10_overall_score_distribution.png
|   |-- 11_key_insights_summary.png
|   |-- 12_confusion_matrix.png
|   |-- 13_feature_importance.png
|   |-- 14_model_comparison.png
|
|-- dashboard/
|   |-- PlacementDashboard.pbix  # Power BI Dashboard file
|
|-- cleaning.py                  
|-- eda.py                      
|-- model.py                     
|-- README.md                    
```

---

## Dataset

- **Source:** Kaggle - College Student Placement Factors Dataset
- **Size:** 10,000 student records
- **Features:**

| Column | Description |
|---|---|
| IQ | Student IQ score |
| Prev_Sem_Result | Previous semester result |
| CGPA | Cumulative Grade Point Average |
| Academic_Performance | Overall academic score |
| Extra_Curricular_Score | Extra curricular activities score |
| Communication_Skills | Communication skills score |
| Projects_Completed | Number of projects completed |
| Internship_Experience | Whether student had internship |
| Placement | Target variable (Placed / Not Placed) |

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Dashboard | Microsoft Power BI |
| IDE | Visual Studio Code |
| Version Control | Git & GitHub |

---

## Data Cleaning (cleaning.py)

Steps performed:
- Dropped unnecessary columns (College_ID)
- Handled missing values (median for numeric, mode for categorical)
- Removed duplicate records
- Standardized column values (Yes/No to Placed/Not Placed)
- Added new columns:
  - Placement_Flag (0/1 for ML model)
  - CGPA_Band (Excellent / Good / Average / Below Average)
  - Overall_Score (weighted combination of all factors)
  - IQ_Band (High / Average / Below Average)

---

## Exploratory Data Analysis (eda.py)

14 charts generated covering:
- CGPA distribution comparison (Box Plot)
- IQ distribution comparison (Box Plot)
- Communication Skills vs CGPA (Scatter Plot)
- Student profile comparison (Horizontal Bar)
- Internship impact analysis (Grouped Bar)
- Feature correlation (Heatmap)
- CGPA trend vs placement rate (Area Chart)
- IQ band placement rate (Horizontal Bar)
- Overall score distribution (Histogram)
- Key insights summary (4-in-1 chart)

---

## Power BI Dashboard

7-page interactive dashboard:

| Page | Content |
|---|---|
| Executive Summary | Single page overview of all key metrics |
| Overview | KPI cards and placement distribution |
| Student Analysis | Internship impact analysis |
| Deep Insights | CGPA vs Overall Score scatter plot |
| IQ Analysis | Placement rate by IQ band |
| CGPA Analysis | Placement rate by CGPA band |
| Filters | Interactive slicers for data exploration |

Download PlacementDashboard.pbix from the dashboard/ folder and open in Power BI Desktop.

---

## Machine Learning Model (model.py)

### Models Trained

| Model | Accuracy |
|---|---|
| Logistic Regression | 90.35% |
| Random Forest | 99.85% |

### Feature Importance (Top Factors)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Communication Skills | 27.7% |
| 2 | IQ | 26.4% |
| 3 | CGPA | 16.9% |
| 4 | Projects Completed | 15.6% |
| 5 | Previous Sem Result | 11.4% |

### Key Finding
Communication Skills is the strongest predictor of placement — more important than CGPA. Students with strong communication skills are significantly more likely to get placed regardless of academic performance.

---

## How to Run

### Step 1 - Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Step 2 - Run Data Cleaning
```bash
python cleaning.py
```

### Step 3 - Run EDA
```bash
python eda.py
```

### Step 4 - Run ML Model
```bash
python model.py
```

### Step 5 - Open Power BI Dashboard
- Open Power BI Desktop
- File -> Open -> Select dashboard/PlacementDashboard.pbix

---

## Key Insights

1. Only 16.59% of students got placed — placement is highly competitive
2. Students with Excellent CGPA (8.5+) have 33.3% placement rate vs 4.4% for Below Average
3. High IQ students (120+) are 5x more likely to get placed than below average IQ students
4. Communication Skills is the #1 placement predictor (27.7% importance)
5. Internship experience alone does not guarantee placement
6. Students with high Overall Score consistently appear in the placed category

---

## Results Summary

```
Dataset          : 10,000 BCA/MCA/CS Students
Placed           : 1,659 (16.59%)
Not Placed       : 8,341 (83.41%)
Average CGPA     : 7.53
Best ML Model    : Random Forest
Model Accuracy   : 99.85%
Top Predictor    : Communication Skills (27.7%)
```

---

## Author

**Name:** Vaishnavi Kenchangoudar 

---

## License

This project is open source and available under the MIT License.
