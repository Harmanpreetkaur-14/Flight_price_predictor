# ✈️ Flight Price Predictor
A end-to-end machine learning web application that predicts Indian domestic flight prices in real time. Built with a Random Forest Regressor trained on 300,000+ flight records, deployed as a live web application using Flask and Render.
# 🔗 Live Demo
👉 https://flight-price-predictor-j4zx.onrender.com/
# 📌 Overview
Flight ticket prices fluctuate based on dozens of factors — airline, travel class, stops, departure time, and how far in advance you book. This project builds a machine learning model that learns these patterns and predicts the fare for any given combination of inputs with an R² score of 0.98.
# 🛠️ Tech Stack
LayerTechnologyMachine Learning = Scikit-learn, Random Forest RegressorData ProcessingPandas, NumPyBackend APIFlask, REST APIDatabaseSQLAlchemy, SQLiteFrontendHTML, CSS, JavaScriptDeploymentGunicorn, Render, GitHub
# 📊 Model Performance

>> Algorithm: Random Forest Regressor
- Training data: 300,000+ Indian domestic flight records
- R² Score: 0.98
- Features used: Airline, travel class, number of stops, departure time slot, arrival time slot, flight duration, days left to departure, source city, destination city

# ⚙️ Features

1) Predict flight prices instantly based on user inputs
2) Stores every prediction in a database with timestamp
3) View last 10 predictions history
4) Booking advice based on days left to departure
5) Price tier classification (Budget / Mid-range / Premium)
6) Fully responsive UI
