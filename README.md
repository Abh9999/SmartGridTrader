# ⚡️ Smart Grid Trading Agent

An AI-powered energy price forecaster built for the Electromobility sector. This project uses Machine Learning (Random Forest) to predict electricity spot market prices, enabling smart charging strategies for EVs.

## 🚀 Project Architecture

* **Orchestration:** n8n (Docker) for automated hourly data fetching.
* **Data Source:** Awattar API (German Spot Market) + Synthetic Data Generator.
* **AI Model:** Scikit-Learn (Random Forest Regressor).
* **Performance:** Achieved **95.8% Accuracy (R² Score)** on historical validation.

## 📂 Structure

* `src/` - Python source code for the AI Brain and Data Simulation.
* `workflows/` - n8n automation workflow JSON files.
* `data/` - Historical CSV/JSON datasets (Gitignored for size).
* `models/` - Saved model binaries.

## 🛠️ How to Run

1.  **Generate Data:** Run the simulation engine.
    ```bash
    python src/generate_prices.py
    ```
2.  **Train the Agent:** Train and evaluate the Random Forest model.
    ```bash
    python src/ai_brain.py
    ```

## 📊 Results

The model successfully identifies daily rush-hour peaks and weekend dips, achieving a Mean Absolute Error (MAE) of < 5% on the test set.