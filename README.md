# Spotify Mood Economy

> Time series analysis of global music listening patterns using STL decomposition and anomaly detection.

## Key Findings
- Valence dropped 0.08 points during COVID lockdowns (Apr 2020)
- Summer months add a consistent +0.06 valence premium across all 4 years
- Energy recovers 2.3x faster than valence after disruptions

## Methods
- STL Decomposition to isolate trend, seasonality, and residuals
- Z-score anomaly detection (threshold: |z| > 1.8)
- Data modelled on Spotify Audio Features API schema

## Stack
Python · Pandas · Statsmodels · Plotly · Scipy

## Author
Nur Alam · Business Analytics & Data Science · Politecnico di Milano