# BDC (PySpark + Streamlit)

This project is a refactor of the original end-to-end notebook into a modular, production-style Python codebase.

## Run

1) Create/activate a Python environment (recommended).

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Start the app:

```bash
streamlit run app.py
```

## Notes

- The app uses **PySpark** for heavy processing and only converts to **Pandas** for plotting via **sampling**.
- Kaggle loading requires the `kaggle` CLI to be installed and authenticated (place `kaggle.json` per Kaggle instructions).
- Saved Spark models are written to the `models/` directory.

