# Pavement Fatigue Cracking Design Tool

A machine learning-based tool for predicting fatigue cracking in flexible pavements using an ensemble of four models.

## Features

- **4-Model Ensemble**: XGBoost, LightGBM, Random Forest, and ANN (PyTorch)
- **Monotonicity Enforcement**: Ensures physically realistic predictions
- **Interactive Visualizations**: Toggle individual models, view prediction ranges
- **Design Comparison**: Compare up to 3 design alternatives
- **Sensitivity Analysis**: Understand parameter impacts
- **Batch Processing**: Upload CSV for multiple predictions
- **Model Confidence Metrics**: Spread-based confidence indicators

## Models

All models were trained using:
- **Training Data**: 260 pavement cases, 62,400 observations
- **Training Method**: Row-level split with 6-fold cross-validation
- **Features**: 19 engineered features including polynomial and Paris Law terms
- **Target Transformation**: Square root transformation with inverse for predictions
- **Validation**: Stratified k-fold CV for robust performance

### Model Performance (6-Fold CV)
- **XGBoost**: R² = 0.94-0.96, RMSE = 3-5%
- **LightGBM**: R² = 0.94-0.96, RMSE = 3-5%
- **Random Forest**: R² = 0.92-0.95, RMSE = 3-5%
- **ANN**: R² = 0.92-0.95, RMSE = 3-5%
- **Ensemble**: R² = 0.94-0.97, RMSE = 3-4%

## Installation

### Prerequisites
```bash
Python 3.8 or higher
```

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd pavement-design-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify model files are present**
```
├── xgboost_model_row_level.pkl
├── lightgbm_model_row_level.pkl
├── random_forest_model_row_level.pkl
├── ann_model_row_level.pkl
└── scaler.pkl
```

4. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Design Input Tab
1. Select design parameters (design life, mix type, structure, traffic)
2. Click "Calculate Prediction"
3. View ensemble prediction with model confidence
4. Toggle individual models on/off to compare
5. Explore sensitivity analysis

### Design Comparison Tab
1. Run a prediction in Design Input first
2. Enter two alternative designs
3. Compare cracking progression curves side-by-side
4. View summary table with design status

### Batch Processing Tab
1. Download CSV template
2. Fill in multiple design cases
3. Upload and run batch predictions
4. Download results with all model predictions

## Model Confidence

The tool provides confidence levels based on model agreement:

- **High Confidence** (Spread < 5%): All models agree closely
- **Medium Confidence** (Spread 5-10%): Reasonable agreement
- **Low Confidence** (Spread > 10%): Significant disagreement, use caution

## Data Ranges

Predictions are most reliable within training data ranges:

| Parameter | Range |
|-----------|-------|
| AC Thickness | 4.0 - 7.0 inches |
| Base Thickness | 8.0 - 24.0 inches |
| Base Modulus | 36.5 - 250 ksi |
| Subgrade Modulus | 5.0 - 20.0 ksi |
| RAP Content | 0 - 30% |

## Design Status Thresholds

- 🟢 **Good**: < 15% cracking - Meets performance criteria
- 🟡 **Acceptable**: 15-30% cracking - Monitor closely
- 🔴 **Early Failure**: > 30% cracking - Consider design modifications

## Technical Details

### Monotonicity Enforcement

All predictions enforce monotonic increase over time:
```python
if year > 0:
    prediction = max(prediction, previous_prediction)
```

This ensures physically realistic behavior (cracking cannot decrease).

### Ensemble Method

Final prediction is the simple average of all 4 models:
```python
ensemble = (xgboost + lightgbm + random_forest + ann) / 4
```

### Feature Engineering

The tool automatically creates 19 features from base inputs:
- 10 base features (thickness, modulus, traffic, etc.)
- 5 polynomial features (Age², ESALs², interactions)
- 4 Paris Law features (fatigue damage accumulation)

## File Structure

```
.
├── app.py                              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── xgboost_model_row_level.pkl         # XGBoost model
├── lightgbm_model_row_level.pkl        # LightGBM model
├── random_forest_model_row_level.pkl   # Random Forest model
├── ann_model_row_level.pkl             # ANN model
└── scaler.pkl                          # Feature scaler
```

## Troubleshooting

### PyTorch Installation Issues (Windows)

If you encounter DLL errors with PyTorch:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues

For large batch processing, process in smaller chunks or reduce design life to minimize memory usage.

### Model Loading Errors

Ensure all 5 .pkl files are in the same directory as app.py.

## Limitations

- Predictions are data-driven and most reliable within training ranges
- Tool aids engineering judgment but does not replace professional expertise
- Confidence metrics indicate model agreement, not absolute accuracy
- Extrapolation beyond training ranges may be unreliable

## License

[Your License Here]

## Citation

If you use this tool in research, please cite:

```
[Your Citation Here]
```

## Contact

[Your Contact Information]

## Acknowledgments

- Models trained on pavement performance database
- Developed using Streamlit, scikit-learn, XGBoost, LightGBM, and PyTorch
