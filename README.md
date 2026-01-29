# Stats Dashboard (Streamlit)

## Run
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py

## Data
Upload a CSV on "01 Data Upload".
Then go through pages:
- Cleaning
- Visualize (Histogram / Pareto)
- Binomial test
- F-test
- T-test
- Time series
- Correlation
- Chi-square

## Notes
- p-value threshold is configurable per page.
- Correlation does NOT imply causation.
