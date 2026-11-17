# Task 2 - remaining cells (sections 4, 5 and 6) for task2.ipynb
# Paste each CELL block into its own new code cell, in this order, BELOW the 'Tuning and compute
# summary' cell (Table 2.2). Markdown blocks are optional: paste them into Markdown cells if you want.

# %% [markdown] ------------- (optional Markdown cell) -------------
# ## 4. Evaluation and visualisation
#
# All four final models are evaluated on **the same held-out test set**: every one of the 433,788 station-days from 2016–2023. The metrics are computed with Spark's `RegressionEvaluator` and with distributed aggregations, so no data is collected to the driver. Only the plots use a fixed sample.
#
# In the formulas, *y* is the actual value, *ŷ* the prediction, *e = y − ŷ* the error, *n* the number of rows and *p* the number of features (54).
#
# | Metric | Definition | What it tells us here |
# |---|---|---|
# | **RMSE** | √(mean(e²)) | Typical error in °C. Large misses count more. This is the deciding metric. |
# | **MAE** | mean(\|e\|) | Average size of the error in °C. Less affected by rare large misses. |
# | MSE | mean(e²) | RMSE squared (°C²); reported because the answer sheet asks for it. |
# | **R²** | 1 − Σe² / Σ(y − ȳ)² | Share of the day-to-day variance explained, compared with always predicting the mean. |
# | Adjusted R² | 1 − (1 − R²)(n − 1)/(n − p − 1) | R² penalised for the 54 inputs. Almost equal to R² when n is this large. |
# | **Explained variance** | 1 − Var(e) / Var(y) | Like R², but it ignores a constant offset. **R² below explained variance means the model is biased.** Spark's `RegressionEvaluator(metricName="var")` returns a different quantity, so this one is computed directly. |
# | **Maximum error** | max(\|e\|) | The single worst prediction in eight test years. |
# | **Inference time** | wall-clock seconds to predict all test rows | Covers preprocessing plus the model. It is the median of 3 runs, forced with Spark's `noop` writer, with the test data cached first so disk reads are not counted. |
#
# Extra diagnostics: bias (mean of e), the share of days predicted within ±3 °C, and the bias on hot days (≥ 32 °C) and on cold days (≤ 0 °C). These are used to read the plots.

# %% ===================== CELL 1 (code) =====================
# ==============================================================================
# 4.1  Metrics on the complete test set + inference time
# REF: Day06_Regression_LiveCoding_Solutions_Colab > cell-9 [Day 6]: RegressionEvaluator(metricName="rmse"/"r2")
#      Changed: all answer-sheet metrics, computed in one distributed pass and cross-checked against
#      RegressionEvaluator; inference timed with the noop writer (median of 3 runs).
## AI: metric aggregation and timing harness written with AI assistance
# ==============================================================================
HOT_C, COLD_C = 32.0, 0.0


def time_inference(model, df, repeats=INFER_REPEATS):
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.transform(df).write.format("noop").mode("overwrite").save()   # EXPLAIN: computes every row, writes nothing
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs), runs


def regression_metrics(pred_df, n_features):
    """All metrics in deg C from one distributed aggregation over every test row."""
    y, e = F.col(TARGET), F.col(TARGET) - F.col("prediction")
    a = pred_df.select(
        F.count(F.lit(1)).alias("n"), F.avg(e * e).alias("mse"), F.avg(F.abs(e)).alias("mae"),
        F.max(F.abs(e)).alias("max_ae"), F.avg(e).alias("bias"),
        F.var_pop(e).alias("var_e"), F.var_pop(y).alias("var_y"),
        F.avg((F.abs(e) <= 3).cast("double")).alias("within3"),
        F.percentile_approx(F.abs(e), 0.95, 10000).alias("p95"),
        F.avg(F.when(y >= HOT_C, e)).alias("bias_hot"), F.sum((y >= HOT_C).cast("int")).alias("n_hot"),
        F.avg(F.when(y <= COLD_C, e)).alias("bias_cold"), F.sum((y <= COLD_C).cast("int")).alias("n_cold"),
    ).first()
    n, r2 = a["n"], 1 - a["mse"] / a["var_y"]
    return {"RMSE": math.sqrt(a["mse"]), "MAE": a["mae"], "MSE": a["mse"], "R2": r2,
            "Adjusted_R2": 1 - (1 - r2) * (n - 1) / (n - n_features - 1),
            "Explained_Variance": 1 - a["var_e"] / a["var_y"], "Max_Abs_Error": a["max_ae"],
            "Bias": a["bias"], "Within_3C_pct": a["within3"] * 100, "P95_Abs_Error": float(a["p95"]),
            "Bias_hot": a["bias_hot"], "Hot_days": int(a["n_hot"]),
            "Bias_cold": a["bias_cold"], "Cold_days": int(a["n_cold"]), "Rows": int(n)}


METRICS, plot_df = {}, None
sample_filter = row_hash % 1000 < PLOT_PER_MILLE                 # the same test days for every model
for name, model in FITTED.items():
    med_s, runs = time_inference(model, test_df)
    preds = model.transform(test_df).select("ID", "date", "TMAX", TARGET, "prediction").cache()
    m = regression_metrics(preds, N_FEATURES)
    for metric, key in (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")):   # cross-check with Spark's evaluator
        spark_val = RegressionEvaluator(labelCol=TARGET, predictionCol="prediction",
                                        metricName=metric).evaluate(preds)
        assert abs(spark_val - m[key]) < 1e-6, f"{name}: {metric} mismatch {spark_val} vs {m[key]}"
    m.update({"Inference_time_s": med_s, "Inference_runs_s": [round(r, 3) for r in runs],
              "Rows_per_s": m["Rows"] / med_s})
    METRICS[name] = m
    samp = (preds.filter(sample_filter).select("ID", "date", "TMAX", TARGET, "prediction").toPandas()
                 .rename(columns={"prediction": f"pred_{name}"}))
    plot_df = samp if plot_df is None else plot_df.merge(samp[["ID", "date", f"pred_{name}"]], on=["ID", "date"])
    preds.unpersist()
    print(f"{name:<24} RMSE {m['RMSE']:.4f} | MAE {m['MAE']:.4f} | R2 {m['R2']:.4f} | "
          f"inference {med_s:.2f} s ({m['Rows_per_s']:,.0f} rows/s)")

# persistence baseline on the same rows: prediction = today's TMAX (no model, no fit)
BASELINE = regression_metrics(test_df.withColumn("prediction", F.col("TMAX")), 1)
plot_df = plot_df.sort_values(["ID", "date"]).reset_index(drop=True)
plot_df.to_csv(RES_DIR / "task2_plot_sample_predictions.csv", index=False)


def comparison_row(name, family, m):
    t = None if name.startswith("Persistence") else m["Inference_time_s"]
    return {"Model": name, "Family": family,
            "RMSE (deg C)": round(m["RMSE"], 4), "MAE (deg C)": round(m["MAE"], 4),
            "MSE (deg C^2)": round(m["MSE"], 4), "R2": round(m["R2"], 4),
            "Adjusted R2": round(m["Adjusted_R2"], 4), "Explained variance": round(m["Explained_Variance"], 4),
            "Max abs error (deg C)": round(m["Max_Abs_Error"], 2), "Bias (deg C)": round(m["Bias"], 3),
            "Within 3 deg C (%)": round(m["Within_3C_pct"], 1),
            "Inference time (s)": round(t, 2) if t is not None else None,
            "Rows per second": round(m["Rows"] / t) if t else None,
            "RMSE vs persistence (%)": round((BASELINE["RMSE"] - m["RMSE"]) / BASELINE["RMSE"] * 100, 1)}


comparison_df = pd.DataFrame([comparison_row(n, MODELS[n]["family"], METRICS[n]) for n in MODELS]
                             + [comparison_row("Persistence baseline", "No model (tomorrow = today)", BASELINE)])
show(comparison_df, f"Table 2.3  Test-set metrics on all {N_TEST:,} test rows ({yr_te[0]}-{yr_te[1]})")
comparison_df.to_csv(RES_DIR / "task2_model_comparison.csv", index=False)
print(f"plot sample: {len(plot_df):,} test days (same days for every model)")

# %% [markdown] ------------- (optional Markdown cell) -------------
# ### 4.2 Actual vs predicted
# There is one panel per model, and all panels use the same scale and the same sample of test days.
#
# - The **dashed line** is a perfect prediction (y = x).
# - The **solid line** is a least-squares trend fitted to the points.
#
# A trend slope below 1 means the predictions are pulled towards typical temperatures: hot days are predicted too cool and cold days too warm. The metrics in each title are from the **complete** test set.

# %% ===================== CELL 2 (code) =====================
# ==============================================================================
# 4.2  Composite figure: actual vs predicted, one panel per model
## AI: figure layout written with AI assistance
# ==============================================================================
import matplotlib.pyplot as plt

POINT, TREND, REF_LINE, GRID_C = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
names = list(MODELS)
y_all = np.concatenate([plot_df[TARGET].to_numpy()] + [plot_df[f"pred_{n}"].to_numpy() for n in names])
lo, hi = np.floor(np.percentile(y_all, 0.1)) - 2, np.ceil(np.percentile(y_all, 99.9)) + 2   # one shared scale

TREND_FITS = {}
ncols = 2
nrows = math.ceil(len(names) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(11, 5.4 * nrows), sharex=True, sharey=True)
for ax, n in zip(axes.flat, names):
    x, yhat = plot_df[TARGET].to_numpy(), plot_df[f"pred_{n}"].to_numpy()
    slope, intercept = np.polyfit(x, yhat, 1)
    TREND_FITS[n] = {"slope": float(slope), "intercept": float(intercept)}
    ax.scatter(x, yhat, s=6, alpha=0.25, color=POINT, linewidths=0, rasterized=True,
               label=f"test days (n = {len(x):,})")
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=REF_LINE, label="perfect prediction (y = x)")
    ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], lw=2, color=TREND,
            label=f"fitted trend: y = {slope:.2f}x {intercept:+.2f}")
    m = METRICS[n]
    ax.set_title(f"{n}  |  {MODELS[n]['family']}\n"
                 f"RMSE {m['RMSE']:.2f} °C  |  MAE {m['MAE']:.2f} °C  |  R² {m['R2']:.3f}",
                 fontsize=10, loc="left")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(color=GRID_C, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
for ax in axes.flat[len(names):]:
    ax.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("Actual next-day TMAX (°C)")
for ax in axes[:, 0]:
    ax.set_ylabel("Predicted next-day TMAX (°C)")
fig.suptitle(f"Fig. 2.1  Actual vs predicted next-day maximum temperature, test years {yr_te[0]}-{yr_te[1]}\n"
             f"Points: the same {len(plot_df):,} sampled test days in every panel; "
             f"metrics: all {N_TEST:,} test rows", fontsize=11.5, x=0.02, ha="left")
fig.tight_layout()
FIG_AVP = FIG_DIR / "task2_actual_vs_predicted.png"
fig.savefig(FIG_AVP, dpi=150, facecolor="white")
plt.show()
print("saved:", rel(FIG_AVP))

# %% [markdown] ------------- (optional Markdown cell) -------------
# ### 4.3 Evidence pack EP2
# This cell answers the answer sheet's EP2 requirement:
#
# - **one loop** prints every model's best parameters and its times;
# - **one composite image** shows actual vs predicted (top row) and residuals vs predicted (bottom row) for each model.
#
# In the residual panels, the black line is the mean residual in 2 °C bins of the prediction. Where it leaves zero, the model is systematically biased for that range of temperatures.

# %% ===================== CELL 3 (code) =====================
# ==============================================================================
# 4.3  EP2 - single loop of best parameters and times + residual and actual-vs-predicted panels
## AI: EP2 layout written with AI assistance
# ==============================================================================
print("=" * 112)
print(f"EP2 | 7006SCN Task 2 | Manish Saini | SID 17142113 | Pool ref 112 | executor: {EXECUTOR_STR}")
print("=" * 112)
print(f"{'Model':<24}{'Best parameters':<38}{'Tuning (s)':>11}{'Final fit (s)':>14}{'Inference (s)':>14}{'RMSE':>9}")
for n, t in TUNED.items():                                             # the single required loop
    bp = ", ".join(f"{k}={v}" for k, v in t["best_params"].items())
    print(f"{n:<24}{bp:<38}{t['tuning_time_s']:>11.1f}{t['final_fit_time_s']:>14.1f}"
          f"{METRICS[n]['Inference_time_s']:>14.2f}{METRICS[n]['RMSE']:>9.4f}")
print(f"{'Persistence baseline':<24}{'(tomorrow = today)':<38}{'-':>11}{'-':>14}{'-':>14}{BASELINE['RMSE']:>9.4f}")
print("=" * 112)

RESID_SPREAD = {}
fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 9.2))
for i, n in enumerate(names):
    x, yhat = plot_df[TARGET].to_numpy(), plot_df[f"pred_{n}"].to_numpy()
    res = x - yhat
    ax = axes[0, i]
    ax.scatter(x, yhat, s=4, alpha=0.25, color=POINT, linewidths=0, rasterized=True)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color=REF_LINE)
    ax.plot([lo, hi], [TREND_FITS[n]["slope"] * lo + TREND_FITS[n]["intercept"],
                       TREND_FITS[n]["slope"] * hi + TREND_FITS[n]["intercept"]], lw=1.8, color=TREND)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_title(f"{n}\nactual vs predicted (R² {METRICS[n]['R2']:.3f})", fontsize=9.5, loc="left")
    ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Predicted (°C)")

    ax = axes[1, i]
    ax.scatter(yhat, res, s=4, alpha=0.25, color=POINT, linewidths=0, rasterized=True)
    ax.axhline(0, ls="--", lw=1.1, color=REF_LINE)
    bins = np.arange(np.floor(lo), np.ceil(hi) + 2, 2.0)
    idx = np.digitize(yhat, bins)
    centres = [(bins[k - 1] + bins[k]) / 2 for k in range(1, len(bins)) if (idx == k).sum() >= 20]
    means = [res[idx == k].mean() for k in range(1, len(bins)) if (idx == k).sum() >= 20]
    ax.plot(centres, means, lw=1.8, color=TREND, label="mean residual (2 °C bins)")
    ax.set_xlim(lo, hi)
    ax.set_title(f"residuals vs predicted (RMSE {METRICS[n]['RMSE']:.2f} °C)", fontsize=9.5, loc="left")
    ax.set_xlabel("Predicted (°C)"); ax.set_ylabel("Residual = actual − predicted (°C)")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    RESID_SPREAD[n] = {lab: float(np.std(res[mask])) for lab, mask in
                       (("below 10 °C", yhat < 10), ("10-25 °C", (yhat >= 10) & (yhat <= 25)), ("above 25 °C", yhat > 25))
                       if mask.sum() >= 20}
    for a in axes[:, i]:
        a.grid(color=GRID_C, lw=0.6); a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
fig.suptitle(f"Fig. EP2  Task 2 diagnostics on {len(plot_df):,} sampled test days ({yr_te[0]}-{yr_te[1]}); "
             f"metrics from all {N_TEST:,} test rows", fontsize=11.5, x=0.01, ha="left")
fig.tight_layout()
FIG_EP2 = FIG_DIR / "task2_ep2_composite.png"
fig.savefig(FIG_EP2, dpi=130, facecolor="white")
plt.show()
print("saved:", rel(FIG_EP2))

# %% [markdown] ------------- (optional Markdown cell) -------------
# ### 4.4 Recommendation and results file
# **Rule, fixed before the results were seen:**
#
# 1. RMSE decides, because large misses are the costly errors for heat and frost warnings.
# 2. Any model within **2%** of the lowest RMSE counts as tied, because a smaller gap is within the run-to-run noise of one train/test split.
# 3. Among tied models, the one with the lower **final fit + inference time** wins, so compute cost is a real constraint.
# 4. The chosen model must beat the persistence baseline.
#
# The results are saved to `task2/results/task2_results.json`. A copy is saved to `results/task2.json`, the file the brief's GitHub checklist reads, next to Task 1's `results/task1.json`.

# %% ===================== CELL 4 (code) =====================
# ==============================================================================
# 4.4  Recommendation rule + results JSON
## AI: recommendation rule and JSON layout written with AI assistance
# ==============================================================================
TOL = 0.02
ranked   = sorted(MODELS, key=lambda n: METRICS[n]["RMSE"])
best_rmse = METRICS[ranked[0]]["RMSE"]
tied     = [n for n in ranked if METRICS[n]["RMSE"] <= best_rmse * (1 + TOL)]
compute  = {n: TUNED[n]["final_fit_time_s"] + METRICS[n]["Inference_time_s"] for n in MODELS}
RECOMMENDED = min(tied, key=lambda n: compute[n])
mae_rank = sorted(MODELS, key=lambda n: METRICS[n]["MAE"])
assert METRICS[RECOMMENDED]["RMSE"] < BASELINE["RMSE"], "No model beats persistence - revisit the features"

RECOMMENDATION = {
    "rule": "lowest test RMSE; models within 2% of the best are tied; ties go to the lowest final fit + "
            "inference time; must beat the persistence baseline",
    "rmse_ranking": ranked, "mae_ranking": mae_rank, "rankings_agree": ranked == mae_rank,
    "tied_within_2pct": tied, "compute_s": {n: round(v, 2) for n, v in compute.items()},
    "recommended_model": RECOMMENDED,
}
print(f"RMSE ranking : {ranked}")
print(f"MAE ranking  : {mae_rank}  ({'agrees' if ranked == mae_rank else 'differs'})")
print(f"tied (<= 2%) : {tied}")
print(f"RECOMMENDED  : {RECOMMENDED} - RMSE {METRICS[RECOMMENDED]['RMSE']:.4f} deg C, "
      f"{(BASELINE['RMSE'] - METRICS[RECOMMENDED]['RMSE']) / BASELINE['RMSE'] * 100:.1f}% below persistence, "
      f"final fit + inference {compute[RECOMMENDED]:.1f} s")

RESULTS = {
    "task": 2, "notebook": "task2/task2.ipynb", "quick_test": QUICK_TEST,
    "student": task1.get("student"), "pool_reference": task1.get("pool_reference"),
    "problem_type": "regression", "target_col": TARGET, "target_meaning": task1["problem"]["target_meaning"],
    "input_contract": "results/task1.json#handoff",
    "rows": {"train": N_TRAIN, "test": N_TEST, "cv_tuning_sample": N_TUNE},
    "features": {"numeric": NUM_FEATURES, "categorical": CAT_FEATURES, "vector_size": N_FEATURES},
    "validation_checks": checks, "preprocessing_check": PREP_CHECK, "target_profile": TARGET_PROFILE,
    "settings": {"seed": SEED, "tune_pct": TUNE_PCT, "final_train_pct": FINAL_TRAIN_PCT, "cv_folds": NUM_FOLDS,
                 "fold_rule": "blocks of consecutive years", "folds": fold_tbl.to_dict("records"),
                 "cv_parallelism": CV_PARALLELISM, "inference_repeats": INFER_REPEATS},
    "executor_config": EXECUTOR_CONFIG,
    "n_algorithms": len(MODELS), "families": sorted({s["family"] for s in MODELS.values()}),
    "n_families": len({s["family"] for s in MODELS.values()}),
    "models": {n: {**{k: v for k, v in TUNED[n].items() if k != "cv_grid"}, "reason": MODELS[n]["reason"],
                   "cv_grid": TUNED[n]["cv_grid"], "test_metrics": METRICS[n], "trend_fit": TREND_FITS[n],
                   "grid_edge": EDGE[n]} for n in MODELS},
    "baseline": {"model": "Persistence (tomorrow = today)", **BASELINE},
    "recommendation": RECOMMENDATION,
    "outputs": sorted(rel(p) for p in list(RES_DIR.glob("*.csv")) + list(FIG_DIR.glob("*.png"))),
}
(RES_DIR / "task2_results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
print("saved:", rel(RES_DIR / "task2_results.json"))

# copy for the brief's checklist (results/task2.json) - never overwrite a file this notebook did not write
brief_json = ROOT / "results" / "task2.json"
if not QUICK_TEST:
    ours = (not brief_json.exists()) or json.loads(brief_json.read_text()).get("notebook") == "task2/task2.ipynb"
    if ours:
        brief_json.write_text(json.dumps(RESULTS, indent=2, default=str))
        print("saved:", rel(brief_json))
    else:
        print(f"NOT overwritten: {rel(brief_json)} was written by another notebook - compare it by hand")

# %% [markdown] ------------- (optional Markdown cell) -------------
# ## 5. Professional process note
# The next cell writes `task2/process_note.md`. The method text is fixed, and **every number in it is read from this run**: the validation table, the missing-value table, the tuning results, the metrics, the trend lines and the timings. If the notebook is run again, the note is rewritten to match the new run.

# %% ===================== CELL 5 (code) =====================
# ==============================================================================
# 5.1  Write task2/process_note.md from this run's measured results
## AI: note template written with AI assistance; every number is filled in from this run's variables
# ==============================================================================
def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def pct_better(n):
    return (BASELINE["RMSE"] - METRICS[n]["RMSE"]) / BASELINE["RMSE"] * 100


def params_txt(p):
    return ", ".join(f"`{k}` = {v}" for k, v in p.items())


ds, best = task1["dataset"], ranked[0]
fold_txt = "; ".join(f"fold {r['fold']}: {r['first_year']}-{r['last_year']} ({r['rows']:,} rows)"
                     for r in fold_tbl.to_dict("records"))
miss_rows = missing_df[missing_df["train_missing_pct"] > 1]
miss_txt = "; ".join(f"`{r.column}` {r.train_missing_pct:.1f}% in train and {r.test_missing_pct:.1f}% in test"
                     for r in miss_rows.itertuples())
shift = [r.column for r in miss_rows.itertuples() if r.test_missing_pct > 1.5 * r.train_missing_pct]
total_fits = sum(t["combinations"] * t["cv_folds"] for t in TUNED.values())
total_tune = sum(t["tuning_time_s"] for t in TUNED.values())
fastest_fit = min(MODELS, key=lambda n: TUNED[n]["final_fit_time_s"])
slowest_fit = max(MODELS, key=lambda n: TUNED[n]["final_fit_time_s"])
fastest_inf = min(MODELS, key=lambda n: METRICS[n]["Inference_time_s"])
slowest_inf = max(MODELS, key=lambda n: METRICS[n]["Inference_time_s"])
gap_model = max(MODELS, key=lambda n: METRICS[n]["Explained_Variance"] - METRICS[n]["R2"])
gap = METRICS[gap_model]["Explained_Variance"] - METRICS[gap_model]["R2"]
tight = max(MODELS, key=lambda n: METRICS[n]["Within_3C_pct"])

tuning_lines = []
for n, t in TUNED.items():
    s = [g["mean_cv_rmse"] for g in t["cv_grid"]]
    spread = max(s) - min(s)
    effect = "barely changed the score" if spread < 0.02 else "made a clear difference"
    edge = (f" The winning value of {', '.join(f'`{k}`' for k in EDGE[n])} is at the edge of the grid, so a "
            f"wider search could still improve it." if EDGE[n] else "")
    tuning_lines.append(
        f"- **{n}**: best {params_txt(t['best_params'])}, mean CV RMSE {t['best_cv_rmse']:.3f} °C "
        f"(std {t['best_cv_rmse_std']:.3f} over the folds). Across its {t['combinations']} combinations the CV RMSE "
        f"ranged from {min(s):.3f} to {max(s):.3f} °C, so the searched settings {effect}.{edge}")

plot_lines = []
for n in MODELS:
    m, tf = METRICS[n], TREND_FITS[n]
    hot = ("under-predicted" if m["Bias_hot"] > 0 else "over-predicted") if m["Bias_hot"] is not None else "n/a"
    cold = ("under-predicted" if m["Bias_cold"] > 0 else "over-predicted") if m["Bias_cold"] is not None else "n/a"
    spread = RESID_SPREAD.get(n, {})
    spread_txt = ", ".join(f"{k}: {v:.2f} °C" for k, v in spread.items())
    plot_lines.append(
        f"- **{n}**: trend slope {tf['slope']:.2f} (1.00 would be perfect); {m['Within_3C_pct']:.1f}% of all test "
        f"days within ±3 °C. Hot days (actual ≥ {HOT_C:.0f} °C, {m['Hot_days']:,} days) were {hot} by "
        f"{abs(m['Bias_hot']):.2f} °C on average, and days at or below {COLD_C:.0f} °C ({m['Cold_days']:,} days) "
        f"were {cold} by {abs(m['Bias_cold']):.2f} °C. Residual standard deviation by predicted range: {spread_txt}.")

note = f"""# Task 2 process note — Modelling at scale

**Student:** Manish Saini (SID 17142113) · **Module:** 7006SCN Machine Learning and Big Data · **Pool ref:** 112
**Notebook:** `task2/task2.ipynb` · **Run finished:** {time.strftime('%Y-%m-%d %H:%M')} · **Mode:** {'QUICK TEST (not results)' if QUICK_TEST else 'full run'}

## 1. Objective and dataset
The objective is to predict tomorrow's maximum temperature at a US weather station (`{TARGET}`, °C) from that station's weather today and on the previous days. This is a regression problem. The data are NOAA GHCN-Daily records (`climate.csv`, {ds['rows']:,} rows × {ds['columns']} columns, {ds['file_size_gib']} GiB). Task 1 turned them into {task1['station_day_rows']:,} station-day rows from a repeatable {H.get('station_sample_pct', 100)}% sample of stations. Task 2 starts from Task 1's output: {N_TRAIN:,} training rows ({yr_tr[0]}–{yr_tr[1]}) and {N_TEST:,} test rows ({yr_te[0]}–{yr_te[1]}), each with {len(NUM_FEATURES)} numeric features and `state_code`.

## 2. Data preparation
- **Loading.** The training and test Parquet files and the fitted preprocessing pipeline were loaded from the paths in `results/task1.json`. The feature list was read from Task 1's pipeline rather than typed by hand.
- **Validation** (`results/task2_validation_checks.csv`):
  - every column had the expected type;
  - the row counts matched Task 1;
  - the training years end before the test years begin;
  - there were no duplicate station-days;
  - the target had no missing values and stayed between {tgt['lo']:.1f} and {tgt['hi']:.1f} °C, inside Task 1's −60 to 60 °C rule.
- **No target leakage.** The label is only built when the next day exists. The lags look back exactly one and two days, and the 7-day means cover today and the six previous days.
- **Missing values** (`results/task2_missing_values.csv`): {miss_txt}. Numeric gaps are filled with the training median, and `state_code` uses `handleInvalid="keep"`. No rows were dropped for a missing target.
- **Feature preparation.** Task 1's five stages (Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler) were rebuilt unfitted with Task 1's settings and placed inside each model's Pipeline. The medians, the state index and the scaling are therefore learned from the training rows of each cross-validation fold, and then from the training years only. When fitted once on all training rows, the rebuilt pipeline reproduced Task 1's {PREP_CHECK['feature_vector_size']}-value feature vector (largest median difference {PREP_CHECK['max_median_difference']:.3g}, largest scaling difference {PREP_CHECK['max_scaler_std_difference']:.3g}).
- **Split.** Task 1's split by year was kept. The test years are the future relative to the training years, as in real forecasting, and they were used once, for the final evaluation.

## 3. Model selection and justification
{md_table(selection_df[['model', 'family', 'reason']])}

The four models increase in flexibility one step at a time: linear, then bagged trees, then explicit pairwise interactions, then boosted trees. The comparison can therefore show what each step adds in accuracy and what it costs in compute. A single decision tree was not used as a separate model, because it is the building block of both tree ensembles and adds no new family.

## 4. Grid search and cross-validation
Each model was tuned with `ParamGridBuilder` and `CrossValidator`: {total_fits} cross-validation fits in total, scored by RMSE. The grids searched the two parameters that control each model's bias–variance trade-off (`results/task2_model_selection.csv`).

Cross-validation used three folds built from blocks of consecutive years ({fold_txt}), on a {TUNE_PCT}% hash sample of the training rows ({N_TUNE:,} rows). Year blocks keep neighbouring, strongly correlated days in the same fold, and they make every fold predict years it was not trained on, as the test set does. Three folds rather than five kept the cost reasonable on a laptop. The best combination of each model was then refitted on all {N_TRAIN:,} training rows. Every combination's score is in `results/task2_cv_results.csv`.

{chr(10).join(tuning_lines)}

## 5. Experimental configuration
{md_table(pd.DataFrame([{'setting': k, 'value': v} for k, v in {**EXECUTOR_CONFIG, 'seed': SEED, 'tuning sample': f'{TUNE_PCT}% of train ({N_TUNE:,} rows)', 'final training rows': f'{N_TRAIN:,}', 'test rows': f'{N_TEST:,}', 'CV folds': f'{NUM_FOLDS} (year blocks)'}.items()]))}

## 6. Training and inference timing methodology
All times are wall-clock seconds measured with `time.perf_counter()` on the machine above.

- **Total tuning time** is the duration of `CrossValidator.fit`. It covers every combination × fold, plus the refit that `CrossValidator` makes on the tuning sample.
- **Final fit time** is the duration of refitting the winning pipeline, preprocessing included, on the full training set. It is reported separately because it is the cost of producing the model that is used.
- **Inference time** is the median of {INFER_REPEATS} runs of predicting all {N_TEST:,} test rows. Each run is forced with Spark's `noop` writer, and the test data was cached first, so disk reads are not counted.

All {len(MODELS)} models together took {total_tune / 60:.1f} minutes of tuning. Times on a single machine can vary by 10–20% between runs, so only clear differences should be read into them.

{md_table(tuning_df[['Model', 'Best parameters', 'Best CV RMSE', 'Total tuning time (s)', 'Final fit time (s)']])}

## 7. Results comparison
All metrics are computed on the complete test set (`results/task2_model_comparison.csv`):

{md_table(comparison_df[['Model', 'RMSE (deg C)', 'MAE (deg C)', 'R2', 'Explained variance', 'Max abs error (deg C)', 'Inference time (s)', 'RMSE vs persistence (%)']])}

- **{best}** had the lowest RMSE ({METRICS[best]['RMSE']:.3f} °C), {pct_better(best):.1f}% below the persistence baseline ({BASELINE['RMSE']:.3f} °C).
- {'Every model beat persistence on RMSE' if all(pct_better(n) > 0 for n in MODELS) else 'Not every model beat persistence on RMSE'}: the RMSE improvements were {', '.join(f'{n} {pct_better(n):.1f}%' for n in ranked)}.
- The RMSE and MAE rankings {'agree' if ranked == mae_rank else 'differ (MAE order: ' + ', '.join(mae_rank) + ')'}, so the choice {'does not depend on the metric' if ranked == mae_rank else 'depends on whether large errors (RMSE) or typical errors (MAE) matter more; RMSE was fixed in advance as the deciding metric because large misses are the costly ones'}.
- {'R² and explained variance are almost identical for every model, so no model has a meaningful systematic bias.' if gap < 0.002 else f'R² is {gap:.3f} below explained variance for {gap_model}, which shows a systematic bias of {METRICS[gap_model]["Bias"]:+.2f} °C on the test years.'}
- The worst single error ranged from {min(METRICS[n]['Max_Abs_Error'] for n in MODELS):.1f} to {max(METRICS[n]['Max_Abs_Error'] for n in MODELS):.1f} °C across the models. These rare large misses are most likely sudden weather changes, such as cold fronts, or extreme records that passed Task 1's validity rule. They affect RMSE more than MAE.
- **Compute:** the final fit ranged from {TUNED[fastest_fit]['final_fit_time_s']:.0f} s ({fastest_fit}) to {TUNED[slowest_fit]['final_fit_time_s']:.0f} s ({slowest_fit}), and inference from {METRICS[fastest_inf]['Inference_time_s']:.1f} s ({fastest_inf}) to {METRICS[slowest_inf]['Inference_time_s']:.1f} s ({slowest_inf}).
- **Recommendation rule** (lowest RMSE; within 2% counts as tied; ties go to the lower final fit + inference time): **{RECOMMENDED}**{' (tied with ' + ', '.join(n for n in tied if n != RECOMMENDED) + ', and cheaper to run)' if len(tied) > 1 else ''}.

## 8. Interpretation of the plots
`figures/task2_actual_vs_predicted.png` shows one panel per model, on the same scale and the same {len(plot_df):,} sampled test days. `figures/task2_ep2_composite.png` adds the residual plots.

{chr(10).join(plot_lines)}

A trend slope below 1 means the predictions are pulled towards typical temperatures. This is expected when a model minimises squared error and part of tomorrow's weather cannot be known from today's. **{tight}** kept the most days within ±3 °C. Where the binned mean residual leaves zero, the model is systematically wrong for that range of temperatures, and this matters most at the hot end, where heat warnings are decided.

## 9. Limitations and reproducibility
**Limitations**
- Task 1 used a {H.get('station_sample_pct', 100)}% sample of stations, so the results describe those stations and states, not the whole US network.
- Tuning ran on a {TUNE_PCT}% sample. This assumes that the ranking of the settings holds on the full training set.
- The year-block folds sometimes validate earlier years with a model trained on later years, so they are not strictly forward in time.
- The test years lie outside the training range of `YEAR_f`. Tree models cannot extrapolate a warming trend; the linear model extrapolates it in a straight line.
- {('Missing values rise in the test years for ' + ', '.join(f'`{c}`' for c in shift) + ', so the imputed medians matter more there.') if shift else 'Missing-value rates are similar in the training and test years.'}
- There is one split and one seed, so the test metrics have no confidence intervals. Differences of a few hundredths of a degree should not be over-read.
- The timings come from one machine in local mode, not from a cluster.

**Reproducibility**
- The seed is fixed ({SEED}).
- The tuning sample and the plot sample are chosen by a hash of `ID` and `date`, so they are the same on any computer.
- The folds come from fixed year blocks, and every setting is declared in cell 0.1.
- Random forest and GBT results can still change slightly with the number of partitions, which depends on the number of cores.
- To reproduce the run: run `notebooks/task1.ipynb`, then open `task2/task2.ipynb` and choose *Restart & Run All*. All tables, figures and this note are rewritten from that run.

## 10. Final conclusion
{RECOMMENDED} is recommended for next-day maximum-temperature prediction on these stations. Its test RMSE was {METRICS[RECOMMENDED]['RMSE']:.3f} °C, against {BASELINE['RMSE']:.3f} °C for persistence ({pct_better(RECOMMENDED):.1f}% better), with R² {METRICS[RECOMMENDED]['R2']:.3f}, a final fit time of {TUNED[RECOMMENDED]['final_fit_time_s']:.0f} s and an inference time of {METRICS[RECOMMENDED]['Inference_time_s']:.1f} s for {N_TEST:,} rows. RMSE was the deciding metric because large misses are the costly errors for heat and frost warnings, and compute cost was used to break ties. These conclusions hold for this station sample, these test years and this single split.

*AI use (Amber): code cells marked `## AI:` and the template of this note were written with AI assistance; all numbers come from my own run of the notebook.*
"""
NOTE_PATH = OUT_DIR / "process_note.md"
NOTE_PATH.write_text(note, encoding="utf-8")
print(f"saved: {rel(NOTE_PATH)} ({len(note.split()):,} words)")

# %% [markdown] ------------- (optional Markdown cell) -------------
# ## 6. Finish
# This releases the cached data and lists every output file. Nothing outside `task2/`, `models/task2/` and `results/task2.json` is written.

# %% ===================== CELL 6 (code) =====================
# ==============================================================================
# 6.1  Release cached data and list the outputs
# ==============================================================================
for df_ in (tune_df, train_df, test_df):
    df_.unpersist()
print("outputs written by this run:")
for p in sorted(list(RES_DIR.glob("*.csv")) + list(RES_DIR.glob("*.json")) + list(FIG_DIR.glob("*.png"))
                + [NOTE_PATH]):
    print(f"  {rel(p):<55} {p.stat().st_size / 1024:8.1f} KB")
print("fitted models:", ", ".join(rel(MODEL_DIR / n) for n in FITTED))
