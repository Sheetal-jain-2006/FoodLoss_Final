# Post-Harvest Food Loss Risk Prediction Using Machine Learning

## 🌟 Project Overview
This project implements a complete, end-to-end Machine Learning research pipeline and interactive Streamlit web application to predict post-harvest food loss risk. Post-harvest loss during transportation and warehouse storage is a significant threat to global food security and supply chain efficiency. By leveraging machine learning models, logistics simulation, and crop-specific rules, this system helps operators classify risk levels and implement tailored mitigation strategies in real-time.

Key components of the system include:
- **AI/ML Spoilage Risk Prediction**: Predicts and classifies post-harvest food loss risk into **Low**, **Medium**, and **High Risk** categories using a serialized machine learning pipeline.
- **Interactive Streamlit Dashboard**: Provides an intuitive interface for real-time predictions, visual analysis, and custom input parameters.
- **Logistics Simulation**: Features preloaded scenario profiles (e.g., *Optimal Cold Chain*, *Transit Delay*, *Environmental Stress*) to demonstrate risk shifts under various transport conditions.
- **Crop-Specific Recommendations**: Includes a rule-based engine that offers custom logistics mitigation advice depending on the crop type and current cargo environment.

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Data Wrangling & Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Ridge Classifier, Decision Trees, Random Forests, Gradient Boosting)
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib
- **Web Interface**: Streamlit

---

## 📁 Project Structure

```text
FoodLoss_Final/
├── data/
│   ├── EuroCrop_agricultural_logistics_dataset.csv  # Raw dataset
│   └── dataset_statistics.csv                       # Statistics of features
├── src/
│   ├── data_loader.py       # Data loading, train-test splitting, target labeling
│   ├── preprocessing.py     # Stateful preprocessing pipeline (reconstruction, OHE, scaling)
│   ├── train.py             # Model training and pipeline serialization
│   ├── evaluate.py          # Model evaluation, metric saving, and report plotting
│   └── recommendations.py   # Crop/logistics recommendations generation
├── notebooks/
│   └── eda.ipynb            # Jupyter notebook for exploratory data analysis
├── models/
│   ├── best_model.joblib          # Selected classifier model (Gradient Boosting)
│   ├── metadata.joblib            # Thresholds, medians, modes, and training stats
│   └── food_loss_pipeline.joblib   # Serialized Pipeline (Preprocessor + Classifier)
├── reports/
│   ├── class_distribution.png     # Class count plot
│   ├── feature_distributions.png  # Histograms of raw vs. log-transformed features
│   ├── correlation_heatmap.png    # Pearson correlation matrix heatmap
│   ├── feature_importance.png     # Visual plot of feature importances
│   ├── confusion_matrix.png       # Evaluation confusion matrix on test set
│   ├── model_comparison.csv       # Training metrics for all candidate models
│   ├── feature_importance.csv      # Importance scores CSV
│   └── metrics.json               # Test set performance metrics
├── app.py                   # Streamlit web dashboard
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (this file)
```

---

## 🎯 Key Features

- **Spoilage Risk Prediction**: Evaluates cargo metrics to classify risk levels into *Low*, *Medium*, or *High* using a serialized pipeline trained on target quantiles.
- **Interactive Risk Calculator**: Real-time evaluation input widgets within the dashboard to allow user-defined logistics profiles and immediately obtain a risk classification.
- **Logistics Simulator**: Built-in scenario presets (e.g., *Optimal Cold Chain*, *Transit Delay*, *Environmental Stress*) that allow users to simulate real-world logistics disruptions and observe their impact on spoilage risks.
- **Dashboard Visualization**: Clear visual representations of prediction scores, historical metrics, model comparison performance plots, and feature importance charts to assist in decision-making.
- **Recommendation Engine**: A crop-specific rule-based engine that maps risk thresholds and raw conditions to specific mitigation suggestions (e.g., adjusting temperature, priority routing, custom warehouse treatments for crops like potatoes, wheat, or tomatoes).

---

## 🛠️ Module Reference & Links

- **Data Loader**: [data_loader.py](src/data_loader.py) - Loads raw data, applies unstratified split (80-20) on the continuous `Spoilage_Risk` variable, and applies 33rd and 66th percentile target labeling derived strictly from the training partition to prevent leakage.
- **Preprocessor**: [preprocessing.py](src/preprocessing.py) - Replaces infinite values, applies `np.log1p` to reconstruct raw scales, clips outliers to physical bounds, imputes missing values using training medians/modes, one-hot encodes categorical variables, and standardizes numeric columns.
- **Train Utility**: [train.py](src/train.py) - Evaluates Dummy, Ridge, Decision Tree, Random Forest, and Gradient Boosting models, saving the best classifier and metadata.
- **Evaluation Utility**: [evaluate.py](src/evaluate.py) - Evaluates the serialized pipeline on the test set, computing accuracy, macro precision, recall, F1, and saving visual reports.
- **Mitigation Advisor**: [recommendations.py](src/recommendations.py) - Custom rule engine translating raw cargo metrics and model outputs into crop-specific recommendations.
- **Jupyter Notebook**: [eda.ipynb](notebooks/eda.ipynb) - Jupyter notebook mapping the data scaling analysis, distributions, correlations, and class thresholds.
- **Web App**: [app.py](app.py) - Interactive web interface deploying the full prediction pipeline.

---

## 🧬 Scientific Insights & Data Quality Findings

1. **Feature Exponentiated Scales (Reconstruction)**:
   In the raw dataset, the numeric features are scaled exponentially (values up to $10^{308}$) and contain infinite values. This makes standard scikit-learn models overflow. 
   We resolved this by applying a natural log transform ($x = \ln(y)$) to reverse the exponentiation. This successfully recovers standard, physically realistic agricultural metrics:
   - **Storage Temperature**: $5\text{--}18^\circ\text{C}$
   - **Storage Humidity**: $60\text{--}90\%$
   - **Warehouse Storage Time**: $5\text{--}30\text{ days}$
   - **Route Distance**: $100\text{--}1000\text{ km}$

2. **Model Performance & Predictive Association**:
   Our rank and linear correlation matrices computed on raw features show very low correlation (Pearson coefficient $< 0.01$) due to the non-linear exponential scaling.
   However, after applying the logarithmic reconstruction transform within the stateful preprocessing pipeline, the underlying predictive associations are successfully recovered.

   The candidate models were evaluated on the test set, yielding the following performance:
   - **Dummy Baseline Classifier**: ~33% accuracy (predicts random categories matching the baseline rate)
   - **Ridge Classifier**: ~67% accuracy
   - **Decision Tree Classifier**: ~63% accuracy
   - **Random Forest Classifier**: ~75% accuracy
   - **Gradient Boosting Classifier (Best Model)**: **~81% accuracy** (Macro F1: ~0.81)

   This highlights the critical necessity of domain-specific data reconstruction (log scaling) to unlock predictive signals in agricultural logistics datasets.

---

## 🏆 Best Model
The selected best performing model for this project is the **Gradient Boosting Classifier**. 
- It achieves a final test set accuracy of **~81%**.
- It is saved as a serialized pipeline in `models/food_loss_pipeline.joblib` and as an individual model in `models/best_model.joblib`.
- It integrates seamlessly with the Streamlit dashboard to provide real-time predictions and decision functions for risk estimation.

---

## 🚀 Running the Project

Follow these steps to set up and run the post-harvest food loss prediction pipeline locally:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/FoodLoss_Final.git
cd FoodLoss_Final
```

### 2. Create a Virtual Environment
Create a Python virtual environment to manage dependencies:
```bash
python -m venv venv
```

### 3. Activate the Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Requirements
Install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Train the Model
Run the training script to evaluate candidate models, select the best model, and serialize the pipeline and metadata:
```bash
python src/train.py
```

### 6. Evaluate the Model
Evaluate the serialized pipeline on the test set and generate performance reports and visualizations:
```bash
python src/evaluate.py
```

### 7. Launch Streamlit
Start the interactive Streamlit web dashboard:
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements
- **Integration of Real-Time IoT Sensors**: Connect the model directly to live MQTT/HTTP telemetry feeds from temperature/humidity sensors during transit.
- **Deep Learning / Advanced Architectures**: Experiment with neural networks or custom loss functions that prioritize penalizing false negatives for "High Risk" cases.
- **Geospatial Optimization Routing**: Integrate weather APIs and maps to dynamically reroute cargo when transit delays or extreme heat are forecasted.
- **Supply Chain Cost Optimization**: Add cost-benefit analysis metrics inside the Streamlit app to weigh the financial cost of mitigation steps against potential crop losses.

---

## 👥 Team
- Sheetal Jain
- Sneha
- Sumrita
- Tanisha
