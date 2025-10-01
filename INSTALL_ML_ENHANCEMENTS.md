# 🚀 Quick Start: Enhanced ML Features

**Time to install:** 5-10 minutes
**Difficulty:** Easy

---

## Step 1: Install TensorFlow (Recommended)

```bash
# Activate your virtual environment
source venv/bin/activate

# Install TensorFlow and Keras
pip install tensorflow>=2.15.0 keras>=3.0.0

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

**Note:** TensorFlow is ~500 MB. The system works without it (fallback mode), but LSTM predictions won't be available.

---

## Step 2: First Run - Initialize Models

```bash
# Run ML analysis for the first time
# This will train models on your historical data
strava-super ml-analysis

# Expected output:
# 🤖 Advanced ML Training Analysis
# Loading 90 days of training data...
# ✅ Loaded 113 days of data
#
# 📈 LSTM Trajectory Forecasting
# Training LSTM model (this may take a minute)...
# ✅ LSTM model trained successfully
#    • Training loss: 0.0234
#    • Validation loss: 0.0289
#    • Epochs: 45
# ...
```

**First run takes:** 30-90 seconds (model training)
**Subsequent runs:** <5 seconds (loaded from disk)

---

## Step 3: Integrate into Daily Workflow

```bash
# Your normal command now includes ML insights!
strava-super run --plan-days 7

# New sections appear:
# - 🤖 Enhanced ML Analysis
# - 📈 LSTM Trajectory Forecast
# - 🔍 Anomaly Detection
```

---

## Step 4: Optional - Export Results

```bash
# Export ML analysis to JSON
strava-super ml-analysis --export ml_results_$(date +%Y%m%d).json

# File created: ml_results_20251001.json
```

---

## Troubleshooting

### If TensorFlow installation fails:

**Option A: Use conda (recommended for Mac/Linux)**
```bash
conda install -c conda-forge tensorflow keras
```

**Option B: Use system without TensorFlow**
```bash
# Everything works except LSTM (uses fallback)
strava-super ml-analysis

# Output will show:
# ⚠️ LSTM Forecast (fallback mode - need 60+ days for training)
```

### If "Insufficient data" error:

```bash
# You need at least 30 days of data
# Solution: Import more history from Strava

strava-super sync --days 90  # Sync last 90 days
```

---

## Quick Test

```bash
# Verify everything works
strava-super ml-analysis --days 60 --forecast-days 7

# Should show:
# ✅ LSTM model trained successfully
# ✅ No anomalies detected
# Analysis complete!
```

---

## What's Next?

1. **Daily use:** `strava-super run --plan-days 7`
2. **Weekly deep-dive:** `strava-super ml-analysis`
3. **Monitor predictions:** Track accuracy over time
4. **Act on anomalies:** Follow recommendations when detected

---

**Installation complete!** 🎉

See `ML_ENHANCEMENTS_GUIDE.md` for detailed documentation.
