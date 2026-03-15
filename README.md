# ⚡ AI Smart Grid Stabilizer

An **AI-powered smart grid simulation system** that predicts renewable energy generation (solar & wind), analyzes electricity demand, and automatically decides grid actions such as battery usage, EV charging policy, and dynamic energy pricing.

The system combines **machine learning models (XGBoost)** with a **real-time interactive dashboard** to demonstrate how modern smart grids can balance supply and demand.

---

# 🧠 System Overview

The project simulates a **24-hour energy grid** using multiple realistic scenarios.

Renewable energy generation is predicted using trained ML models and the grid controller dynamically reacts to energy imbalances.

### Grid Decisions Include

* Battery charging / discharging
* Power rerouting
* Emergency grid stabilization
* EV charging policy adjustment
* Dynamic electricity pricing

---

# ⚙️ Features

### ⚡ Renewable Energy Prediction

* Solar power prediction using **XGBoost**
* Wind power prediction using **XGBoost**
* Uses environmental features such as:

  * Solar irradiance
  * Cloud cover
  * Temperature
  * Wind speed
  * Wind direction
  * Air density

### 🔋 Smart Battery Management

Battery automatically responds to grid imbalance:

| Condition        | Action                  |
| ---------------- | ----------------------- |
| Supply >> Demand | Store energy in battery |
| Balanced grid    | No action               |
| Supply < Demand  | Use backup battery      |
| Severe deficit   | Emergency support       |

---

### 🚗 EV Charging Intelligence

The grid dynamically controls EV charging based on grid conditions:

* ⚡ Fast Charging (surplus energy)
* 🚗 Normal Charging
* ⚠ Limited Charging
* 🛑 Charging Paused during emergency

---

### 💰 Dynamic Pricing

Electricity price multiplier adjusts automatically:

| Grid Condition | Price             |
| -------------- | ----------------- |
| Surplus energy | Cheap electricity |
| Balanced grid  | Normal price      |
| Energy deficit | High price        |

---

### 📊 Interactive Dashboard

The web dashboard provides:

* Real-time KPIs
* Supply vs demand charts
* Renewable energy generation graphs
* Battery status visualization
* EV charging policy display
* Grid action alerts
* 24-hour simulation log
* Raw ML model input features

---

# 🧩 Grid Simulation Patterns

Each simulation run cycles through **7 grid scenarios**:

| Pattern   | Scenario                         |
| --------- | -------------------------------- |
| Pattern 1 | Supply < Demand (battery backup) |
| Pattern 2 | Supply > Demand (energy storage) |
| Pattern 3 | Balanced grid                    |
| Pattern 4 | Extreme demand spike             |
| Pattern 5 | High EV charging load            |
| Pattern 6 | Night wind surplus               |
| Pattern 7 | Full day ramp (mixed conditions) |

---

# 🛠 Technology Stack

### Backend

* Python
* Flask
* XGBoost
* NumPy
* Pandas
* Joblib

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

### Machine Learning

* XGBoost regression models
* Feature scaling with sklearn scalers
* Time-series lag features

---

# 📁 Project Structure

```
SmartGrid
│
├── app.py                     # Flask backend
├── index.html                 # Dashboard UI
│
├── solar_power_model.json     # Solar prediction model
├── wind_power_model.json      # Wind prediction model
│
├── solar_scaler_y.pkl
├── wind_scaler_y.pkl
│
├── solar_features.pkl
├── solar_cap_max.pkl
│
├── combined_demand_model.json
├── combined_features.pkl
├── combined_scaler_y.pkl
│
├── wind_model.json
├── wind_scaler_y.pkl
│
├── aep_profile.pkl
│
├── grid_simulation_results.csv
├── smart_grid_results.csv
│
└── README.md
```

---

# 🚀 Running the Project

## 1️⃣ Install dependencies

```bash
pip install flask flask-cors xgboost numpy pandas joblib
```

---

## 2️⃣ Run the backend server

```bash
python app.py
```

Server will start at:

```
http://localhost:5000
```

---

## 3️⃣ Open the dashboard

Open in browser:

```
http://localhost:5000
```

Click:

```
▶ Run Simulation
```

to generate a **24-hour smart grid simulation**.

---

# 🔬 Machine Learning Models

The project uses **trained XGBoost models** for renewable energy prediction.

### Solar Model Inputs

* GHI (Global Horizontal Irradiance)
* DNI / DHI
* Cloud cover
* Solar elevation
* Panel efficiency
* Temperature
* Historical power lag features

---

### Wind Model Inputs

* Wind speed
* Wind direction
* Air density
* Wind turbine capacity
* Historical wind speed lag features

---

# 📡 Backend API Endpoints

### Run Simulation

```
GET /api/simulate
```

Returns:

* 24-hour grid data
* battery levels
* EV policy
* grid actions
* price multiplier
* alerts

---

### System Status

```
GET /api/status
```

Returns:

* model loading status
* battery capacity
* next simulation pattern

---

### Reset Simulation Counter

```
POST /api/reset
```

---

# 📊 Example Output Data

Each hour produces:

```
hour
solar_kw
wind_kw
supply_kw
demand_kw
imbalance_kw
battery_level
grid_action
ev_policy
price_multiplier
```

---

# 🎯 Purpose of the Project

This project demonstrates how **AI can autonomously stabilize renewable energy grids** by:

* predicting renewable generation
* forecasting energy demand
* controlling storage and EV loads
* dynamically adjusting pricing

Such systems are essential for **future smart grids powered by renewable energy**.

---

