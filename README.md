# Credit Default Risk Predictor — Streamlit App

A simple form-based UI for the XGBoost credit default model trained in
`credit_default_prediction_cleaned.ipynb`. Enter a borrower's details and
get a predicted default probability plus a risk flag based on your
cost-optimized decision threshold.

## Folder structure required

Everything lives in **one flat folder** — no subfolders:

```
your_app_folder/
├── app.py
├── requirements.txt
├── xgb_model.json
├── xgb_model.pkl
├── feature_names.json
├── model_metadata.json
└── preprocessing_params.json
```

These artifact files are produced by Section 11 of the notebook
("Save Model & Dependencies") — copy them into the same folder as `app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually `http://localhost:8501`).

## Deploy for free (Streamlit Community Cloud)

1. Push all the files above (flat, same folder) to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Point it at your repo/branch and set the main file to `app.py`.
4. Deploy — you'll get a shareable public URL.

## What the app does

- Loads the trained XGBoost model and its saved threshold/metadata from
  the same directory as `app.py` (cached with `st.cache_resource` so it
  only loads once per session).
- Collects the 10 borrower features via number inputs / a slider.
- Builds a single-row dataframe in the exact column order the model
  expects (using `feature_names.json`, so column-order bugs can't happen).
- Runs `predict_proba`, compares against the cost-optimized threshold
  saved in `model_metadata.json`, and shows a risk verdict, probability,
  and a progress bar.
- Sidebar shows model performance (ROC-AUC, PR-AUC), the threshold, the
  cost assumptions behind it, and the best hyperparameters.
