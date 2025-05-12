# 🏥 CMS Hospital Rating Prediction (3-Class Model)
This project uses publicly available CMS (Centers for Medicare &amp; Medicaid Services) data to build a 3-class classification model that predicts the quality rating of acute care hospitals in the United States. The final model is built using Random Forest, with class balancing via SMOTE and explanation using DALEX


## 📌 Project Objectives

- Transform raw hospital data into a clean, structured dataset.
- Perform exploratory data analysis to uncover patterns in ratings.
- Engineer features that meaningfully describe hospital performance.
- Build a multi-class classification model (Low / Medium / High rating).
- Balance classes with SMOTE and explain the model using DALEX.

---

## 🧰 Tools & Libraries

Built entirely in **RStudio**, using:

- `tidyverse`, `caret`, `randomForest`, `smotefamily`, `DALEX`, `janitor`
- Visualization: `ggplot2`, `DT`
- Model explanation: `DALEX`

---

## 📁 Dataset

- **Source**: [CMS Hospital General Information](https://data.cms.gov/provider-data/dataset/xubh-q36u)
- **Focus**: Acute Care Hospitals only
- **Target Variable**: `hospital_overall_rating`, transformed into:
  - `Low` (1-2 stars)
  - `Medium` (3 stars)
  - `High` (4-5 stars)

---

## 📊 Key Analysis Steps

### 1. Data Cleaning
- Removed rows with missing or "Not Available" ratings
- Filtered only **Acute Care Hospitals**
- Added region labels based on state

### 2. Exploratory Data Analysis (EDA)
- Ratings analyzed by **state**, **ownership**, and **region**
- Correlation heatmaps to detect predictor relationships
- Listing top-rated hospitals

### 3. Feature Engineering
Created meaningful features:
- `readm_safety_gap`: difference in readmission and safety metrics
- `mort_ratio`, `safety_ratio`: performance ratios

### 4. Modeling
- Used `caret` to train a **Random Forest** classifier
- Balanced classes using **SMOTE** to improve learning

### 5. Evaluation
- Confusion matrix on test set
- Feature importance visualization
- Model explanation using **DALEX**

---

## 📈 Results

- Model accurately classified hospitals into `Low`, `Medium`, or `High` groups.
- Key drivers: `readm_safety_gap`, ownership type, mortality and safety ratios.
- Balanced dataset after SMOTE improved prediction of underrepresented classes.
- - ✅ Accuracy: 48.17%
- While this accuracy may not seem high at first glance, it's important to note that this is a three-class classification problem (Low, Medium, High), which is inherently more complex than binary classification.
- The model performs better than random chance (33.3%) and is effective in distinguishing performance tiers across hospitals, especially after SMOTE class balancing.
- Key drivers of prediction include:
  - `readm_safety_gap` (readmission vs. safety issues)
  - `mort_ratio` and `safety_ratio` (normalized quality indicators)
  - `hospital_ownership` and `state`

---

## 📌 To Run Locally

1. Clone the repository
2. Download the CMS dataset and save as `Hospital_General_Information.csv`
3. Open and run the RMarkdown file (`CMS_Hospital_3Class_Model.Rmd`) in RStudio
4. Install missing packages if prompted

---

## 🙌 Acknowledgements

- CMS for open-access data
- DALEX & caret package authors for modeling tools
- Inspiration from public healthcare data science initiatives

---

## 🚀 Future Enhancements

- Deploy as a Shiny App for live prediction
- Add more granular patient outcome features
- Include cost and demographic data for holistic modeling


