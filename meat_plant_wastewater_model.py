"""
Meat plant wastewater synthetic data + linear models

What this script does:
1) Generates a daily synthetic dataset calibrated to manager-provided anchors:
   - Weekday wastewater ≈ 2.1 MGD; weekend wastewater ≈ 0.6 MGD.
   - Water used ≈ wastewater / fraction (fraction ~0.93–0.98).
   - Nominal 114 gal/hog (used to derive hogs/day from water use after accounting for cleaning).
   - 3 shifts/day (24h) with variable uptime.
   - Water temperature "mix" features: fractions of cold / 140F / 180F.
   - Continuous cleaning_intensity instead of discrete cycles.

2) Fits LinearRegression and Ridge models to predict wastewater_volume:
   - production-only feature set (no water_used to avoid leakage)
   - with water_used (for comparison)

3) Prints metrics and interpretable coefficients, and produces visualizations:
   - Scatter: kills_per_hour vs wastewater (colored by weekend)
   - Partial effect curve for kills_per_hour

Notes on data consistency:
- Given water_used ~2.1–2.2 MGD on weekdays and 114 gal/hog, the implied hogs/day is ~18–19k
  (before subtracting cleaning/base). This conflicts with earlier 20,500/day. In this script,
  we prioritize manager-provided flows and let hogs/day be derived accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------
# Configuration and Assumptions
# ---------------------------

@dataclass
class PlantConfig:
    seed: int = 42
    n_days: int = 200

    # Manager anchors
    weekday_wastewater_mean: float = 2.10e6  # gal/day
    weekend_wastewater_mean: float = 0.60e6  # gal/day
    wastewater_sd_weekday: float = 7.0e4
    wastewater_sd_weekend: float = 6.0e4

    # Wastewater fraction (wastewater = fraction * water_used)
    ww_fraction_weekday_mu: float = 0.96
    ww_fraction_weekend_mu: float = 0.95
    ww_fraction_sd: float = 0.015
    ww_fraction_bounds: Tuple[float, float] = (0.90, 0.99)

    # Per-hog water (nominal). We will derive hogs/day from water_used and this factor.
    water_per_hog_gal: float = 114.0

    # Cleaning/base water components we subtract before dividing by per-hog factor
    cleaning_gal_weekday_mu: float = 1.0e5     # rough order; adjusts "free" water not tied to hogs
    cleaning_gal_weekend_mu: float = 1.6e5     # relatively higher proportion on weekends
    cleaning_gal_sd: float = 2.0e4
    base_ops_gal_mu: float = 4.0e4             # hoses, hand-wash stations, misc
    base_ops_gal_sd: float = 1.0e4

    # Shifts and uptime
    shift_hours: float = 24.0
    uptime_weekday_mu: float = 0.95
    uptime_weekend_mu: float = 0.80
    uptime_sd: float = 0.03
    uptime_bounds: Tuple[float, float] = (0.6, 1.0)

    # Line efficiency (captures minor speed variation not explained by uptime)
    line_eff_mu: float = 0.98
    line_eff_sd: float = 0.03
    line_eff_bounds: Tuple[float, float] = (0.9, 1.05)

    # Water temperature mix (fractions sum to 1). Use Dirichlet with mean ~ [0.60, 0.25, 0.15]
    temp_mix_alpha: Tuple[float, float, float] = (6.0, 2.5, 1.5)

    # Production intensity noise
    hogs_noise_sd: float = 300.0

    # BOD/COD/TSS anchors and sensitivities
    bod_mean: float = 1130.0
    cod_mean: float = 2500.0
    tss_mean: float = 300.0

    bod_sd: float = 90.0
    cod_sd: float = 140.0
    tss_sd: float = 60.0

    # Sensitivity to production intensity (p = hogs_day / weekday_mean_hogs)
    bod_prod_coef: float = 180.0
    cod_prod_coef: float = 300.0
    tss_prod_coef: float = 60.0

    # Sensitivity to cleaning intensity and hot water usage
    cod_cleaning_coef: float = 50.0
    tss_cleaning_coef: float = 35.0
    tss_hot180_coef: float = 20.0    # hotter water can lift fats/oils -> suspended solids

    # Bounds
    bod_bounds: Tuple[float, float] = (600.0, 4000.0)
    cod_bounds: Tuple[float, float] = (1000.0, 8000.0)
    tss_bounds: Tuple[float, float] = (80.0, 1200.0)


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


# ---------------------------
# Synthetic data generator
# ---------------------------

def generate_synthetic_data(cfg: PlantConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range("2024-01-01", periods=cfg.n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    is_weekend = (dow >= 5).astype(int)

    # Wastewater fraction
    ww_frac = rng.normal(
        loc=np.where(is_weekend == 1, cfg.ww_fraction_weekend_mu, cfg.ww_fraction_weekday_mu),
        scale=cfg.ww_fraction_sd
    )
    ww_frac = _clip(ww_frac, *cfg.ww_fraction_bounds)

    # Wastewater volume (gal/day), anchored to manager values
    ww_mu = np.where(is_weekend == 1, cfg.weekend_wastewater_mean, cfg.weekday_wastewater_mean)
    ww_sd = np.where(is_weekend == 1, cfg.wastewater_sd_weekend, cfg.wastewater_sd_weekday)
    wastewater = rng.normal(ww_mu, ww_sd)
    wastewater = _clip(wastewater, 0.4e6, 2.4e6)

    # Water used per day derived from wastewater and fraction
    water_used = wastewater / ww_frac

    # Cleaning intensity (continuous), weekends slightly higher proportionally
    cleaning_intensity = _clip(
        rng.normal(loc=np.where(is_weekend == 1, 1.1, 1.0), scale=0.1, size=cfg.n_days),
        0.7, 1.4
    )

    cleaning_gal = rng.normal(
        loc=np.where(is_weekend == 1, cfg.cleaning_gal_weekend_mu, cfg.cleaning_gal_weekday_mu) * cleaning_intensity,
        scale=cfg.cleaning_gal_sd
    )
    cleaning_gal = _clip(cleaning_gal, 2.0e4, 3.0e5)

    base_ops_gal = _clip(rng.normal(cfg.base_ops_gal_mu, cfg.base_ops_gal_sd, cfg.n_days), 1.0e4, 1.0e5)

    # Compute hogs/day from water balance: water_used ≈ per_hog * hogs + cleaning + base
    # => hogs ≈ (water_used - cleaning - base) / per_hog
    hogs = (water_used - cleaning_gal - base_ops_gal) / cfg.water_per_hog_gal
    # Bound hog counts (weekday ~ 16k–20k, weekend lower)
    hogs_lo = np.where(is_weekend == 1, 0.0, 14000.0)
    hogs_hi = np.where(is_weekend == 1, 14000.0, 20000.0)
    hogs = _clip(hogs + rng.normal(0, cfg.hogs_noise_sd, cfg.n_days), hogs_lo, hogs_hi)

    # Uptime and line efficiency
    uptime = _clip(
        rng.normal(loc=np.where(is_weekend == 1, cfg.uptime_weekend_mu, cfg.uptime_weekday_mu),
                   scale=cfg.uptime_sd, size=cfg.n_days),
        *cfg.uptime_bounds
    )
    line_eff = _clip(rng.normal(cfg.line_eff_mu, cfg.line_eff_sd, cfg.n_days), *cfg.line_eff_bounds)

    # Kills per hour derived from hogs/day
    effective_hours = cfg.shift_hours * uptime
    kills_per_hour = np.divide(hogs, np.maximum(effective_hours * line_eff, 1e-6))
    kills_per_hour = _clip(kills_per_hour, 100.0, 1300.0)

    # Water temperature mix (fractions sum to 1): cold, 140F, 180F
    temp_mix = rng.dirichlet(cfg.temp_mix_alpha, size=cfg.n_days)
    frac_cold = temp_mix[:, 0]
    frac_140 = temp_mix[:, 1]
    frac_180 = temp_mix[:, 2]

    # Production intensity relative to weekday mean (computed from weekday days only)
    weekday_mask = (is_weekend == 0)
    weekday_mean_hogs = max(hogs[weekday_mask].mean() if weekday_mask.any() else hogs.mean(), 1.0)
    prod_intensity = hogs / weekday_mean_hogs

    # BOD/COD/TSS around manager anchors with small dependence on production and cleaning
    bod = rng.normal(cfg.bod_mean + cfg.bod_prod_coef * (prod_intensity - 1.0),
                     cfg.bod_sd, cfg.n_days)
    bod = _clip(bod, *cfg.bod_bounds)

    cod = rng.normal(cfg.cod_mean
                     + cfg.cod_prod_coef * (prod_intensity - 1.0)
                     + cfg.cod_cleaning_coef * (cleaning_intensity - 1.0),
                     cfg.cod_sd, cfg.n_days)
    cod = _clip(cod, *cfg.cod_bounds)

    tss = rng.normal(cfg.tss_mean
                     + cfg.tss_prod_coef * (prod_intensity - 1.0)
                     + cfg.tss_cleaning_coef * (cleaning_intensity - 1.0)
                     + cfg.tss_hot180_coef * (frac_180 - 0.15),
                     cfg.tss_sd, cfg.n_days)
    tss = _clip(tss, *cfg.tss_bounds)

    df = pd.DataFrame({
        "date": dates,
        "is_weekend": is_weekend,
        "shift_hours": np.full(cfg.n_days, cfg.shift_hours),
        "uptime": uptime,
        "line_eff": line_eff,
        "cleaning_intensity": cleaning_intensity,
        "frac_cold": frac_cold,
        "frac_140F": frac_140,
        "frac_180F": frac_180,
        "kills_per_hour": kills_per_hour,
        "hogs_killed_per_day": hogs.astype(int),
        "water_used_per_day": water_used,
        "wastewater_volume": wastewater,
        "BOD_mg_L": bod,
        "COD_mg_L": cod,
        "TSS_mg_L": tss,
        # For diagnostics: implied per-hog water excluding cleaning/base
        "implied_gal_per_hog": np.divide(
            np.maximum(water_used - cleaning_gal - base_ops_gal, 0.0),
            np.maximum(hogs, 1e-6)
        ),
        "cleaning_gal_component": cleaning_gal,
        "base_ops_gal_component": base_ops_gal
    })
    return df


# ---------------------------
# Modeling utilities
# ---------------------------

def fit_and_evaluate(df: pd.DataFrame, include_water_used: bool, ridge_alpha: float = 1.0) -> Dict:
    # Production-only avoids leakage (no water_used_per_day)
    base_features = [
        "hogs_killed_per_day",
        "kills_per_hour",
        "shift_hours",
        "uptime",
        "cleaning_intensity",
        "frac_140F",
        "frac_180F",
        "is_weekend",
    ]
    features = base_features + (["water_used_per_day"] if include_water_used else [])
    X = df[features].copy()
    y = df["wastewater_volume"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    def make_pipe(model):
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model)
        ])

    lin = make_pipe(LinearRegression())
    ridge = make_pipe(Ridge(alpha=ridge_alpha, random_state=7))

    lin.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    def eval_model(name, model):
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"{name:<18} | RMSE: {rmse:,.0f} gal/day | R^2: {r2:.3f}")

    print("\nModel performance (predicting wastewater_volume):")
    eval_model("LinearRegression", lin)
    eval_model("Ridge", ridge)

    # De-standardize ridge coefficients for interpretability
    scaler: StandardScaler = ridge.named_steps["scaler"]
    model: Ridge = ridge.named_steps["model"]
    coef = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.sum(scaler.mean_ * model.coef_ / scaler.scale_)

    coef_df = pd.DataFrame({"feature": features, "slope_gal_per_unit": coef}).sort_values(
        "slope_gal_per_unit", key=np.abs, ascending=False
    )
    print("\nApprox. marginal effects (Ridge; gal/day per unit):")
    with pd.option_context("display.float_format", "{:,.1f}".format):
        print(coef_df.to_string(index=False))
    print(f"Intercept (gal/day): {intercept:,.0f}")

    return {
        "features": features,
        "lin": lin,
        "ridge": ridge,
        "coef_df": coef_df,
        "intercept": intercept,
        "splits": (X_train, X_test, y_train, y_test)
    }


def visualize(df: pd.DataFrame, model_info: Dict):
    sns.set(style="whitegrid", context="talk")

    # Scatter: kills/hour vs wastewater (color by weekend)
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df, x="kills_per_hour", y="wastewater_volume",
        hue=df["is_weekend"].map({0: "Weekday", 1: "Weekend"}),
        palette="Set1", alpha=0.8
    )
    plt.title("Kills per hour vs Wastewater volume")
    plt.xlabel("Kills per hour")
    plt.ylabel("Wastewater volume (gal/day)")
    plt.tight_layout()
    plt.show()

    # Partial effect of kills/hour with others at median
    features = model_info["features"]
    ridge = model_info["ridge"]
    med = df[features].median(numeric_only=True)

    kph_grid = np.linspace(df["kills_per_hour"].quantile(0.05),
                           df["kills_per_hour"].quantile(0.95), 60)
    X_partial = pd.DataFrame({f: np.full_like(kph_grid, med[f], dtype=float) for f in features})
    X_partial["kills_per_hour"] = kph_grid
    y_partial = ridge.predict(X_partial)

    plt.figure(figsize=(8,6))
    plt.plot(kph_grid, y_partial, color="crimson", lw=3)
    plt.title("Partial effect: kills/hour -> predicted wastewater (others at median)")
    plt.xlabel("Kills per hour")
    plt.ylabel("Predicted wastewater (gal/day)")
    plt.tight_layout()
    plt.show()


def diagnostics(df: pd.DataFrame):
    print("\nDiagnostics on internal consistency:")
    weekday = df[df.is_weekend == 0]
    weekend = df[df.is_weekend == 1]
    print(f"Weekday mean wastewater (gal): {weekday.wastewater_volume.mean():,.0f}")
    print(f"Weekend mean wastewater (gal): {weekend.wastewater_volume.mean():,.0f}")
    print(f"Weekday mean water used (gal): {weekday.water_used_per_day.mean():,.0f}")
    print(f"Weekend mean water used (gal): {weekend.water_used_per_day.mean():,.0f}")
    print(f"Weekday mean implied gal/hog: {weekday.implied_gal_per_hog.mean():.1f}")
    print(f"Weekend mean implied gal/hog: {weekend.implied_gal_per_hog.mean():.1f}")
    print(f"Weekday mean hogs/day: {weekday.hogs_killed_per_day.mean():,.0f}")
    print(f"Weekend mean hogs/day: {weekend.hogs_killed_per_day.mean():,.0f}")


def main():
    cfg = PlantConfig()
    df = generate_synthetic_data(cfg)

    print("First 5 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.head().to_string(index=False, formatters={
            "water_used_per_day": lambda x: f"{x:,.0f}",
            "wastewater_volume": lambda x: f"{x:,.0f}",
            "kills_per_hour": lambda x: f"{x:,.1f}",
            "implied_gal_per_hog": lambda x: f"{x:,.1f}"
        }))

    diagnostics(df)

    print("\n--- Production-only model (no water_used to avoid leakage) ---")
    info_no_leak = fit_and_evaluate(df, include_water_used=False, ridge_alpha=2.0)

    print("\n--- Model with water_used (illustrates value of metered flow) ---")
    info_with_water = fit_and_evaluate(df, include_water_used=True, ridge_alpha=1.0)

    # Visualize using the production-only model info
    visualize(df, info_no_leak)


if __name__ == "__main__":
    main()
