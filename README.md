# Arsenal Forecast

A small football forecasting project for Arsenal EPL and UCL title race title race.

It does two things:

- trains a simple match outcome model from historical EPL data
- simulates Arsenal vs Man City title outcomes from the remaining 2025/26 fixtures
- simulates Arsenal vs Athletic UCL semi final outcomes from the remaining 2025/26 fixtures

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn joblib requests python-dotenv
pip install -e .
```

## Run

Prepare training data:

```bash
python3 src.data.prepare_training_data
```

Train the model:

```bash
python3 src.models.match_predictor
```

Build current-season strength:

```bash
python3 src.features.current_season_strength
```

Run the title simulation:

```bash
python3 scripts/simulate_epl_title.py
```

## Required Files

- `data/raw/epl_historical_2022_2024.csv`
- `data/raw/arsenal_epl_2025_26.csv`
- `data/raw/mancity_epl_2025_26.csv`
- `data/raw/epl_full_2025_26.csv`

## Outputs

- `outputs/models/match_predictor.pkl`
- `outputs/models/team_strength.csv`
- `outputs/models/current_season_strength.csv`

## Optional API Fetching

If you want to fetch data from API-Football, add a `.env` file:

```env
API_FOOTBALL_KEY=your_api_key_here
```

Then run:

```bash
python3 scripts/fetch_historical_seasons.py
python3 scripts/fetch_current_season.py
```

## Notes

- The title simulator is currently hardcoded for Arsenal vs Man City.
- `scripts/train_model.py`, `scripts/run_forecast.py`, and the YAML config files are placeholders.
- Use `python3`, in linux environment.
- Use `python`, in windows environment.
