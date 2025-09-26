# Vaayudut

**Short-term Forecasting of Air Pollutants in India**  

Vaayudut provides **state-wise forecasts of ground-level air pollutants (O₃ and NO₂)** using a hybrid approach that combines **satellite data (Sentinel-5P)**, **reanalysis datasets**, and **ground monitoring stations**. 
Designed for urban air quality management, it leverages **time-series forecasting** and **deep learning models** to deliver actionable insights.

---

## Need

Air pollution is a pressing challenge in rapidly urbanizing Indian cities. Ground-level O₃ and NO₂ significantly impact human health, ecosystems, and climate. Vaayudut aims to **predict short-term pollutant levels** to inform policy makers, researchers, and citizens for better decision-making.

---

## Project Goals

- Predict **daily O₃ and NO₂ concentrations** across Indian states.  
- Integrate **satellite, reanalysis, and ground-based data** for accurate forecasting.  
- Provide an **interactive dashboard** for visualization and analysis.

---

## Data Sources & Preprocessing

- **Satellite Data:** Sentinel-5P (TROPOMI) for O₃ and NO₂  
- **Reanalysis Data:** Copernicus / NASA for meteorological parameters  
- **Ground Data:** OpenAQ, CPCB for real measurements  

**Preprocessing Steps:**  
1. Data cleaning and missing value handling  
2. Temporal alignment and resampling  
3. Normalization / scaling for model input  

---

## Model / Approach

- **Time-Series Forecasting:** LSTM / GRU models for sequential prediction  
- **Feature Engineering:** Incorporate meteorological variables, previous pollutant levels  
- **Model Evaluation:** MAE, RMSE, and R² metrics  

---

## Features

- Multi-source Data Integration  
- State-wise Forecasts  
- Trend Analysis & Hotspot Identification  
- Interactive Streamlit Dashboard  
- Extensible for additional pollutants or regions  

---

## Technologies Used

- Programming Languages: Python
  
- Frameworks & Libraries:
1. Streamlit – for building interactive dashboards
2. Pandas & NumPy – data manipulation and numerical computations
3. Matplotlib & Seaborn – data visualization
4. TensorFlow – machine learning and deep learning models

- Data Sources / APIs:
1. Sentinel-5P API – satellite-based atmospheric pollutant data
2. OpenAQ API – ground-level pollutant measurements
3. Copernicus / NASA Reanalysis – meteorological and environmental data

- Tools / Environment:
1. Google Collab & VS Code for development
2. Git & GitHub for version control

---

## Usage

1. Open the Streamlit app
2. Select a state to view forecasts.
3. View current and predicted forecasts.
4. Explore visualizations for O₃ and NO₂ trends.

---

## Future Scope

- Extend to include PM2.5, CO, SO₂ forecasting
- Scalable to other Indian states
- Real-time API integration for live updates
- Deploy on cloud for interactive public dashboards
- Explore ensemble models for higher accuracy
