# LSTM Migration Guide

## Summary of Changes

This document describes the migration from XGBoost to LSTM with Attention mechanism for stock price forecasting.

---

## Key Changes

### 1. **Model Architecture** (`src/agents/model_agent.py`)

**Before:** XGBoost with Calibrated Classifier
- Tree-based model
- No sequence handling
- SHAP TreeExplainer for interpretability

**After:** LSTM with Attention Mechanism
- **Architecture:**
  - 2-layer LSTM (128 hidden units)
  - Attention mechanism for temporal importance
  - Dense layers (64 → 32 → 1)
  - Dropout layers (0.2) for regularization
  
- **Key Features:**
  - Sequence length: 60 days
  - Batch size: 64
  - Learning rate: 0.001
  - Epochs: 50
  - Binary Cross-Entropy loss
  - Adam optimizer

- **Input Shape:** `(batch_size, 60, num_features)`
- **Output:** Binary probability (0-1)

### 2. **Explainability** (`src/agents/explainer_agent.py`)

**Before:** SHAP TreeExplainer
- Feature importance from tree splits
- Shapley values

**After:** Attention-based Explainability
- Uses attention weights to show temporal importance
- Feature importance calculated from:
  - Average attention weights over recent timesteps
  - Feature magnitude
- More interpretable for time-series data

### 3. **Data Processing** (`src/agents/data_agent.py`)

**Fixed:** Timezone handling
- Now properly handles "IST" timezone in dates
- Converts to UTC then removes timezone to avoid warnings

### 4. **Dependencies** (`requirements.txt`)

**Removed:**
- `xgboost==2.1.1`
- `shap==0.46.0`

**Added:**
- `torch==2.1.0` (PyTorch for deep learning)

### 5. **Training Pipeline** (`src/pipeline/train.py`)

**Changes:**
- Model initialization with LSTM config
- Updated model saving to PyTorch format (`.pth`)
- Modified prediction pipeline to handle sequences
- Updated explainer to use attention mechanism
- Report title changed to "LSTM with Attention — Top Picks"

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Activate your virtual environment first
.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install PyTorch (CPU version)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

**Note:** For GPU support, install PyTorch with CUDA:
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Run Training

```bash
python -m src.pipeline.train
```

---

## Model Outputs

### Saved Artifacts

**Location:** `./models/`

1. **`lstm_model.pth`** - PyTorch model checkpoint containing:
   - Model state dictionary
   - Feature scaler (StandardScaler)
   - Feature names
   - Model configuration

2. **`meta.json`** - Feature metadata

---

## How LSTM Works in This Project

### 1. **Sequence Creation**
- Takes tabular data and creates sliding windows of 60 days
- Each sequence contains 60 timesteps with all features
- Predicts the next day's direction (up/down)

### 2. **Training Process**
```
Input: [60 days × features] → LSTM Layers → Attention → Dense Layers → Probability
```

### 3. **Attention Mechanism**
- Learns which timesteps are most important for prediction
- Provides interpretability: "Which days in the 60-day window mattered most?"
- Weights are normalized using softmax

### 4. **Prediction**
- For each stock, uses last 60 days to predict next day
- Returns probability of upward movement (0-1)
- Attention weights show temporal importance

---

## Architecture Diagram

```
Input Sequence (60 days × features)
           ↓
    LSTM Layer 1 (128 units)
           ↓
    LSTM Layer 2 (128 units)
           ↓
    Attention Mechanism
    (learns temporal importance)
           ↓
    Context Vector (weighted average)
           ↓
    Dense Layer (64 units) + ReLU + Dropout
           ↓
    Dense Layer (32 units) + ReLU + Dropout
           ↓
    Dense Layer (1 unit) + Sigmoid
           ↓
    Probability (0-1)
```

---

## Key Differences: XGBoost vs LSTM

| Aspect | XGBoost | LSTM with Attention |
|--------|---------|---------------------|
| **Input Type** | Tabular (1 row = 1 prediction) | Sequential (60 rows = 1 prediction) |
| **Temporal Awareness** | None (treats each row independently) | Yes (learns patterns across time) |
| **Explainability** | SHAP (feature importance) | Attention weights (temporal importance) |
| **Training Time** | Fast (~minutes) | Slower (~30-60 minutes) |
| **Memory Usage** | Low | Higher (requires sequences) |
| **Best For** | Tabular data, feature interactions | Time-series, temporal patterns |
| **Overfitting Risk** | Medium (tree depth control) | Higher (needs dropout, regularization) |

---

## Configuration Options

You can adjust LSTM parameters in `src/pipeline/train.py`:

```python
model_agent = ModelAgent(ModelAgentConfig(
    sequence_length=60,      # Number of days to look back
    hidden_size=128,         # LSTM hidden units
    num_layers=2,            # Number of LSTM layers
    dropout=0.2,             # Dropout rate
    learning_rate=0.001,     # Adam learning rate
    batch_size=64,           # Training batch size
    epochs=50                # Training epochs
))
```

---

## Expected Performance

### Training Output
```
Training on device: cpu
Created X sequences of length 60
Epoch [10/50], Loss: 0.XXXX
Epoch [20/50], Loss: 0.XXXX
...
Eval: {'accuracy': 0.XX, 'f1_up': 0.XX, 'brier': 0.XX}
```

### Report Generation
- PDF report: `./reports/daily_report.pdf`
- Top 3 stock picks with:
  - Probability of upward movement
  - Conviction level
  - Attention-based explanations
  - Stop-loss and target prices

---

## Troubleshooting

### Issue: "Import torch could not be resolved"
**Solution:** Install PyTorch:
```bash
pip install torch==2.1.0
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU:
```python
ModelAgentConfig(batch_size=32)  # Reduce from 64
```

### Issue: Slow training
**Solutions:**
1. Reduce epochs: `epochs=30`
2. Reduce sequence length: `sequence_length=30`
3. Use GPU if available

### Issue: Poor accuracy
**Solutions:**
1. Increase epochs: `epochs=100`
2. Adjust learning rate: `learning_rate=0.0001`
3. Add more data (longer historical period)
4. Tune hyperparameters

---

## Future Enhancements

1. **Multi-stock LSTM:** Train separate models per stock
2. **Bidirectional LSTM:** Learn from past and future context
3. **Attention Visualization:** Plot attention weights over time
4. **Ensemble:** Combine LSTM with other models
5. **Online Learning:** Update model with new data
6. **Multi-horizon:** Predict 1, 3, 5 days ahead

---

## References

- PyTorch Documentation: https://pytorch.org/docs/
- LSTM Attention Paper: "Attention Is All You Need"
- Stock Prediction with LSTM: Various research papers on arXiv

---

**Last Updated:** November 5, 2025
