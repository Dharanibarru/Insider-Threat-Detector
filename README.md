# Sentinel – Insider Threat Prediction from Login/Session Behaviour
Behavioural-biometric anomaly detection (per-user z-scores + Isolation Forest) on CMU CERT-format logs.

- `index.html` – static dashboard (no build step, no backend). Includes an in-browser detector: load a sample or upload your own CSV (logon logs or daily features, optional answer key); analysis runs on the user's device.
- `data/` – synthetic CERT-format logs + real CERT-derived samples (MIT repos manas765/insider-threat-detection, sunilmc197/Insider-Threat-Behavioral-Intelligence-System; original data: CMU CERT r4.2, Glasser & Lindauer 2013)
- `pipeline/` – Python code: generate_data → feature_engineering → anomaly_detection

## Run locally
    python3 -m http.server 8000      # open http://localhost:8000
## Re-run the pipeline
    pip install -r requirements.txt
    cd pipeline && python3 generate_data.py && python3 feature_engineering.py && python3 anomaly_detection.py
## Deploy
GitHub Pages (push to main, Settings > Pages > Source: GitHub Actions) · Netlify (`netlify deploy --prod --dir=.`) · Docker (`docker build -t sentinel . && docker run -p 8080:80 sentinel`)
