# Meat Plant Wastewater: Synthetic Data + Linear Models

This package generates a calibrated synthetic dataset and fits multivariable linear models to predict daily wastewater volume for a pork processing facility.

## Why this design

Manager-provided anchors:
- Water used per day: ~2.1–2.2 MGD (weekdays)
- Wastewater per day: ~2.1 MGD (weekdays), ~0.6 MGD (weekends)
- Water per hog: ~114 gal/hog (average)
- BOD: ~1130 mg/L, COD: ~2500 mg/L, TSS: ~300 mg/L
- 3 shifts/day × 8 hours = 24 hours; cleaning is continuous
- Water temperatures: cold / 140°F / 180°F (varies by process)

These anchors are treated as constraints the synthetic generator must satisfy on average.

Note on inconsistency:
- 114 gal/hog × 20,500 hogs/day ≈ 2.34 MGD, which exceeds 2.1–2.2 MGD.
- In this synthetic generator, we prioritize the manager’s flow numbers and allow weekdays to settle around ~18–19k hogs/day. The script prints “implied gal/hog” so you can see how closely the synthetic data matches 114 once cleaning/base components are removed.

## Generative model (mechanistic + calibration)

Notation:
- WW: wastewater_volume (gal/day)
- WU: water_used_per_day (gal/day)
- f: wastewater fraction (WU -> WW), ~0.93–0.98
- H: hogs_killed_per_day
- kph: kills_per_hour
- CI: cleaning_intensity (continuous index)
- U: uptime fraction (0.6–1.0)
- Tmix: (frac_cold, frac_140F, frac_180F), Dirichlet

Steps:
1. WW ~ Normal(2.1e6, 7e4) on weekdays and Normal(0.6e6, 6e4) on weekends, clipped to reasonable bounds.
2. WU = WW / f, where f ~ Normal(0.96 wkd, 0.95 wknd), clipped [0.90, 0.99].
3. CI ~ Normal(1.0 weekdays, 1.1 weekends) with bounds [0.7, 1.4].
4. Cleaning water ≈ CI × (100k weekdays, 160k weekends) with noise.
5. Base ops water ≈ Normal(40k, 10k).
6. Hogs/day derived from balance:
   H ≈ (WU − Cleaning − Base) / 114, then bounded (weekday 14k–20k, weekend ≤14k).
7. Uptime ~ Normal(0.95 weekdays, 0.80 weekends) in [0.6, 1.0]; line_eff ~ Normal(0.98, 0.03).
8. kph = H / (24 × Uptime × line_eff), bounded [100, 1300].
9. Tmix ~ Dirichlet(6, 2.5, 1.5) → fractions cold/140F/180F.
10. Production intensity p = H / mean(H on weekdays).
11. BOD ~ Normal(1130 + 180 × (p − 1), 90), clipped [600, 4000].
12. COD ~ Normal(2500 + 300 × (p − 1) + 50 × (CI − 1), 140), clipped [1000, 8000].
13. TSS ~ Normal(300 + 60 × (p − 1) + 35 × (CI − 1) + 20 × (frac_180F − 0.15), 60), clipped [80, 1200].

This keeps relationships plausible while honoring the manager’s weekday/weekend flow anchors.

## Feature sets and leakage

- Production-only model (recommended for real-time prediction without meters):
  - Features: hogs_killed_per_day, kills_per_hour, shift_hours, uptime, cleaning_intensity, frac_140F, frac_180F, is_weekend
  - Target: wastewater_volume
  - No water_used_per_day included → avoids leakage

- With water_used_per_day:
  - Adds water_used_per_day to features
  - Much higher R² (as expected) because WW ≈ f × WU, but this depends on having reliable metering.

## How to run

```bash
python meat_plant_wastewater_model.py
```

The script prints:
- A head sample
- Diagnostics (weekday/weekend means, implied gal/hog, hogs/day)
- Metrics (RMSE, R²) for LinearRegression and Ridge
- Coefficient table (gal/day per unit) for interpretability
- Two plots:
  - Scatter of kills/hour vs wastewater (colored by weekend)
  - Partial effect curve for kills/hour

## Interpreting coefficients (Ridge)

- Positive slope on hogs_killed_per_day or kills_per_hour indicates higher production increases wastewater.
- is_weekend typically has a large negative slope (weekend wastewater is lower).
- cleaning_intensity is mildly positive (more cleaning → more wastewater).
- frac_140F/frac_180F usually have small effects on volume (more relevant to load/energy than flow).

## How to adapt to your real data

1) Replace synthetic data:
   - Load your CSV with columns you measure (hogs/day, kph, uptime or shift hours, cleaning proxy, temperature mix if available, water_used_per_day if metered, wastewater, BOD/COD/TSS).
   - Pass that dataframe into the same modeling pipeline (train/test split, pipelines, metrics).

2) Guard against leakage:
   - If you won’t know water_used in real time, train without it and track the accuracy gap.

3) Time-aware split:
   - Prefer the first 75% of days for training and the last 25% for testing to mimic real deployment.

4) Diagnostics to add (optional):
   - Residual vs. fitted plot
   - Rolling R² to detect drift
   - VIF (variance inflation factor) to assess collinearity (kph ≈ hogs/day / hours).

## Operational recommendations (high ROI)

- Instrumentation:
  - Ensure reliable flow meters on incoming water and final effluent; log at 1–5 min resolution.
  - Create a lightweight cleaning_intensity signal (PLC tag for hose stations/CIP pumps, or a short manual log until automated).
  - Track uptime (downtime minutes per shift); expose to your historian/CSV export.

- Water reduction:
  - Target per-hog water (114 gal/hog) by station; pilot low-cost hose nozzles/auto-shutoffs.
  - Reuse final-rinse water for pre-rinse where permitted; measure change in per-hog gallons and COD.

- Load management:
  - If weekends have lower flows but higher cleaning proportion, monitor COD/TSS peaks and pre-dose or equalize to avoid surcharge spikes.
  - Build a daily dashboard: predicted vs. actual wastewater, per-hog water, and surcharge drivers (BOD/COD/TSS).

- Data feedback loop:
  - Compare model predictions to meters weekly; investigate days with largest residuals to find process anomalies or sensor issues.

## Limitations

- Synthetic data is for pipeline development and teaching only; coefficients will look reasonable because the generator encodes the mechanism.
- Replace each synthetic feature with a measured signal as it becomes available, retrain, and compare shifts.

## Next steps

- Share a week or two of real data (even partial). We’ll:
  - Swap out the generator, keep the pipeline, and produce a concise model card.
  - Add residual diagnostics and a simple dashboard-ready export (CSV with predictions and errors).
