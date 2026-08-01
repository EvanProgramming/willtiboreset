# Phase 2 Web Dashboard Report

## 1. Completed Pages

WillTiboReset now has a single-page React dashboard deployed to GitHub Pages. The page contains the following sections:

| Section | Content |
|---------|---------|
| **Header** | Project title, subtitle, "System Online" status indicator, and last update time |
| **Prediction Hero** | Large-format probabilities for reset within 5h, 24h, and 48h, each with a horizontal indicator bar |
| **AI Reasoning** | Main factors driving the prediction plus the model's natural-language reasoning |
| **Evidence Sources** | Latest Tibo, OpenAI, and Community signals sourced from `data/tweets.json` |
| **Prediction Timeline** | Pure SVG line chart of 24-hour probability history from `data/prediction_history.json` |
| **Model Performance** | Total predictions, resolved predictions, accuracy, Brier score, calibration error, and calibration bins |
| **About Model** | Flat process-flow diagram: RSS Sources → LLM Signal Analysis → Bayesian Evidence Model → Prediction → Calibration |

The UI follows a strict minimal-tech aesthetic: pure black background (`#000000`), cyan accent (`#00f0ff`), white text, no gradients, no glassmorphism, no heavy shadows, and no circular progress bars.

## 2. Technologies Used

- **Framework**: React 18
- **Build Tool**: Vite 5
- **Language**: JavaScript (JSX)
- **Styling**: Plain CSS with CSS variables
- **Charts**: Custom SVG (no external chart library)
- **Deployment**: GitHub Pages via GitHub Actions
- **Package Manager**: npm

No additional UI libraries or animation frameworks were added, keeping the bundle small and the design fully controllable.

## 3. Data Connection Method

The dashboard loads data at runtime via `fetch`, using paths relative to the configured Vite `base`:

```
/data/prediction.json
/data/model_performance.json
/data/prediction_history.json
/data/tweets.json
```

During the build step, `scripts/copy-data.js` copies the latest JSON files into `public/data/` so they are included in the static output:

- `output/prediction.json` → `public/data/prediction.json`
- `output/model_performance.json` → `public/data/model_performance.json`
- `data/prediction_history.json` → `public/data/prediction_history.json`
- `data/tweets.json` → `public/data/tweets.json`

This keeps the frontend decoupled from the Python pipeline: the prediction system continues to write to its original paths, and only the dashboard build step moves the data into the static site.

## 4. GitHub Pages Deployment Method

The existing `.github/workflows/predict.yml` was extended to build and deploy the dashboard after every prediction run:

1. Run `python update_model.py`
2. Run `python predict.py` (generates `output/prediction.json`, `output/model_performance.json`, and updates `data/prediction_history.json`)
3. Set up Node.js 20 and run `npm ci`
4. Run `npm run copy-data` to populate `public/data/`
5. Run `npm run build` to produce `dist/`
6. Commit the updated JSON files (`output/prediction.json`, `output/model_performance.json`, `data/model_state.json`, `data/prediction_history.json`)
7. Upload `dist/` as a Pages artifact and deploy to GitHub Pages

Workflow permissions were updated to include `pages: write` and `id-token: write`. A separate `deploy` job depends on the `predict` job and uses `actions/deploy-pages@v4`.

## 5. Verification Results

- `npm install` succeeded
- `npm run build` succeeded and produced `dist/` with all required assets and data files
- Local preview rendered all seven dashboard sections with live data
- Python test suite passed: **133 passed**
- Changes were committed and pushed to `main`

## 6. Phase 1 Compatibility

No Phase 1 prediction logic was modified. The following files remain unchanged:

- `model/survival_model.py`
- `predict.py`
- `calibration.py`
- `model/data_models.py`
- `config.py`
- `collectors/` and `analyzer/` directories
- `data/reset_history.json` and `data/model_state.json`

Only the workflow, `.gitignore`, and new frontend files were added.

## 7. Future Extensible Features

- Auto-refresh on the page (e.g., every 5 minutes) so open browsers receive updates without a manual reload
- Multi-language support
- Detailed per-bin calibration chart comparing predicted mean vs. actual frequency
- Historical prediction table with actual reset outcomes
- Light/dark theme toggle
- Server-sent notifications when probability crosses a threshold
