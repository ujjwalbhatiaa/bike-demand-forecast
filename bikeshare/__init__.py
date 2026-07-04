"""bikeshare — end-to-end ML pipeline for hourly bike-share demand forecasting.

Modules:
    data      — load & validate the UCI Bike Sharing dataset
    features  — feature engineering (cyclical time encoding, real-unit weather)
    evaluate  — leakage-safe time-based splitting + regression metrics
    models    — the model ladder compared in train.py
"""

__version__ = "0.1.0"
