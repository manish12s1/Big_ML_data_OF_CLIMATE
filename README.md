### **7006SCN2627SEPNOV - Machine Learning and Big Data**

### **Student Information**

\- \- \- \- \- \- \- \- \- \-

#### Student Name: Manish Saini

#### Student ID: 17142113

### Module: 7006SCN - Machine Learning and Big Data

#### **Module Leader**: Dr Katerina Stamou

### **Assessment**: Applied Core Assessment

#### **Dataset**: NOAA Global Historical Climatology Network – Daily (GHCN-Daily), US stations, mirrored to Hugging Face as CatLikeIceCream/temperature

#### **Dataset Source**: Hugging Face

#### **Direct URL**: https://huggingface.co/datasets/CatLikeIceCream/temperature

### Notes / Access Caveats

GHCN-Daily US stations, mirrored to Hugging Face as CatLikeIceCream/temperature. 17.1M rows, 35 columns. Target: temperature or precipitation.

#### **GitHub repository URL**:  https://github.coventry.ac.uk/7006SCN-2627SEPNOV/sainim10_17142113.git

#### Tableau Public URL: https://public.tableau.com/app/profile/manish.saini8855/viz/7006SCNManishSaininext-dayTMAXforecasting/D1Dataqualitypipelinemonitoring



**Big Data problem**  

\- \- \- \- \- \- \- \- \- \-



I forecast tomorrow's maximum temperature (TMAX, °C) at US weather stations, a regression problem that matters for heat warnings and energy planning. The data is NOAA GHCN-Daily (Pool ref 112): 17,097,453 rows × 35 columns in a 2.54 GiB CSV covering 1970–2023, processed in PySpark 4.2 on a 12-core laptop. Task 1 builds the data pipeline, Task 2 tunes and compares four Spark MLlib regressors, Task 3 tunes Spark and explains the models, and Task 4 tells the story in four Tableau dashboards.



**Task1 - Problem, Data and Pipeline**  

### **1.1 Problem, Dataset and Why This Pairing**
Define the real-world continuous outcome problem. State explicitly that this is a REGRESSION ML problem predicting a continuous target variable. Name the dataset allocated to you by Your Module Leader and state its source and argue why this dataset suits this problem. State your target variable and what it means. For the higher bands, identify a limitation of the dataset for this problem.

This is a regression problem. I predict tomorrow's maximum temperature (°C) at a US weather station, which helps with energy planning and heat warnings. My dataset (Pool ref 112) is NOAA GHCN-Daily for US stations, from Hugging Face (CatLikeIceCream/temperature), with 17,097,453 rows and 35 columns from 1970 to 2023. Each station is a long daily record, so lag and seasonal features need no future days. The target, label_TMAX_next, is the next day's TMAX at the same station. A limitation: there is no location or elevation data, and record gaps limit the lag features (Fig. T1.1).

### **1.2 Big Data Justification and Compliance**
*Complete the table using figures from your own dataset. State the licence and note any ethical concerns (privacy, bias, consent).*

| **Dimension**    | **How your dataset demonstrates this dimension** | **Figure / table ref** |
|------------------|--------------------------------------------------|------------------------|
| Volume           | 17,097,453 rows; 2.54 GiB; 21 partitions after loading. | Fig. EP1 |
| Velocity         | Historical batch data; a full scan took 5.34 s on 12 cores. | task1.ipynb §5 |
| Variety          | 132 element codes; 31 wide day columns; mixed units. | Fig. EP1; task1.ipynb §4 |
| Veracity         | DAY1–28 are 14.16–14.60% empty and DAY31 50.29%; 0 duplicate rows. | Fig. T1.1; task1.ipynb §8 |
| Value            | Tomorrow's TMAX has r = 0.935 with today's TMAX. | Table T1.2 |
| **Licence**      | None on the mirror; NOAA data is public domain, cite Menne et al. (2012). | task1.ipynb §1 |
| **Ethical note** | No personal data; states with fewer stations may get higher errors. | Table 3.3b (Task 3) |

*Which of the five V's dominate for this dataset, and why?*

Volume and Veracity. Reshaping the full file was too big for my 16 GB laptop, so I kept 5% of stations. Missing days (Fig. T1.1) break the lags and leave some targets empty.

### **1.3 Data Preparation and PySpark Pipeline**

*Complete the table. Describe each step, justify why it was applied, and name the taught session it comes from.*

| **Stage order** | **Step**     | **Decision taken and justification** | **Transformer used** |
|-----------------|--------------|--------------------------------------|----------------------|
| 1               | Impute       | Median, as PRCP and SNOW are skewed; dropping nulls loses days after gaps (Day 2 workbook). | Imputer |
| 2               | Index state  | handleInvalid='keep' handles states not seen in training (Data Pipelines). | StringIndexer |
| 3               | Encode state | Stops states being read as ordered numbers (Feature Engineering at Scale). | OneHotEncoder |
| 4               | Assemble     | Spark ML needs one features vector (Data Pipelines). | VectorAssembler |
| 5               | Scale        | Unit variance; withMean=False keeps the vectors sparse. | StandardScaler |

*Why must the stages run in this order, and what would break if two were swapped?*

VectorAssembler fails on nulls, so imputing comes first. The encoder needs the StringIndexer output and the scaler needs the assembled vector, so swapping either pair fails at fit time. All stages are fitted on 1970–2015 only; the 2016–2023 test years are never used.

### **EP1 Evidence Pack**
*Paste one image, produced from a single notebook cell, showing in order: SparkSession initialisation, df.printSchema(), row and column count, file size verification, partition count after ingestion, and the fitted pipeline stage list.*

![Fig. EP1: the single EP1 cell output](<figures/EP1_evidence_pack.png>)

*Fig. EP1: one cell (task1.ipynb §11) prints, in order, the SparkSession, df.printSchema(), 17,097,453 rows × 35 columns, the 2.54 GiB file size, 21 partitions after ingestion and the five fitted pipeline stages.*

**Task 1 Process Note**

Writing bronze Parquet failed on Windows because HADOOP_HOME was unset. I installed winutils.exe and hadoop.dll, set HADOOP_HOME before starting Spark, and the 2.54 GiB CSV is now parsed only once. 



**Task2 - Modelling at Scale**

\- \- \- \- \- \- \- \- \- \-

Notebook: [`notebooks/task2.ipynb`](notebooks/task2.ipynb) · results in `task2/results/` · figures in `task2/figures/`

### **2.1 Overview and Justification of the Four ML Algorithms**

| **Algorithm** | **Family** | **Why appropriate for this problem** | **ParamGrid searched** |
|---|---|---|---|
| LinearRegression | Linear | Today's TMAX already has r = 0.935 with tomorrow's, so a linear model is a cheap, strong baseline. | regParam {0, 0.01, 0.1} × elasticNetParam {0, 0.5, 1} |
| RandomForestRegressor | Tree ensemble | Picks up season, snow and state effects with no scaling; trees train in parallel. | numTrees {20, 50} × maxDepth {8, 12} |
| FMRegressor | Kernel (factorisation machine) | Learns pairs such as state × season and still scales to 2.5 M rows. | factorSize {4, 8} × maxIter {50, 100} |
| GBTRegressor | Free choice (boosting) | Each tree fixes the last one's errors; slower, but often the most accurate. | maxDepth {3, 5} × maxIter {30, 60} |

### **2.2 Tuning and Compute Summary Table**

| **Algorithm** | **Hyperparameters (best)** | **CV folds** | **Tuning / final fit time (s)** | **Executor config** |
|---|---|---|---|---|
| LinearRegression | regParam 0, elasticNetParam 0 | 3 | 161 / 49 | local[*], 12 cores, 4 GB driver |
| RandomForestRegressor | numTrees 50, maxDepth 12 | 3 | 717 / 655 | same |
| FMRegressor | factorSize 8, maxIter 100 | 3 | 720 / 1083 | same |
| GBTRegressor | maxDepth 5, maxIter 60 | 3 | 1100 / 662 | same |

I tuned on a 10% sample (255,855 rows) with folds made from blocks of years, then refitted the winner on all 2,555,512 training rows. No regularisation won for Linear Regression, and its whole grid moved only 0.04 °C, so shrinkage was not needed. For the other three models the largest setting always won, at the edge of the grid, so more capacity might still help, but time limited me. Random Forest beat Linear Regression by 0.13 °C in CV but took 13 times longer to fit.

### **2.3 Metrics Comparison Table**

| **Metric / Model** | **LinearRegression** | **RandomForest** | **FMRegressor** | **GBTRegressor** |
|---|---|---|---|---|
| RMSE (°C) | 3.974 | 3.792 | 4.082 | 3.853 |
| MAE (°C) | 2.992 | 2.860 | 3.104 | 2.910 |
| MSE | 15.79 | 14.38 | 16.66 | 14.84 |
| R² | 0.889 | 0.899 | 0.883 | 0.896 |
| Adjusted R² / Explained variance | 0.889 / 0.889 | 0.899 / 0.899 | 0.883 / 0.883 | 0.896 / 0.896 |
| Max error (°C) | 29.1 | 30.1 | 31.7 | 29.0 |
| Inference time (s) | 2.08 | 10.71 | 3.05 | 3.15 |

All four beat the "tomorrow = today" baseline (RMSE 4.397). I use RMSE because big misses on very hot or cold days matter most. Random Forest is best, but GBT is only 1.6% worse, which is inside my 2% tie rule. GBT predicts three times faster, so I recommend GBT.

### **EP2 Evidence Pack**

![EP2: best parameters and times for all four models, with actual-vs-predicted and residual plots](<task2/figures/task2_ep2_composite.png>)

*Fig. EP2: one loop prints each model's best parameters and times, above actual-vs-predicted and residual plots for the same 5,143 test days. This is regression, so there are no ROC or confusion-matrix panels.*

**Task 2 Process Note**

Random Forest crashed with Java OutOfMemoryError: maxMemoryInMB=512 across 12 tasks and 2 parallel fits exceeded the 4 GB heap. I reproduced it, set 64 MB, and cross-validation completed.



**Task3 - Evaluation, Optimisation and Explainability**

\- \- \- \- \- \- \- \- \- \-

Notebook: [`notebooks/task3.ipynb`](notebooks/task3.ipynb) · results in `task3/results/` and `results/task3.json` · figures in `task3/figures/`

### **3.1 Spark UI Interpretation and Optimisation**

| **Change made** | **Setting** | **Rationale** | **Stage** | **Before (s)** | **After (s)** |
|---|---|---|---|---|---|
| Persist scored test set | MEMORY_AND_DISK, 4 columns | 6 of 8 jobs re-scored all test rows | test scan + score | 20.4 | 7.4 |
| Repartition before caching | 11 → 24 partitions | 11 uneven partitions, 12 cores | GBT retrain passes | 103.6 | 126.6 |
| Parallel CV fits | parallelism 1 → 2 | cores only 48% busy | CrossValidator.fit | 98.3 | 94.4 |

The baseline timeline (Table 3.1b) showed recomputation: 93% of stage time was the same scan and 60-tree scoring, repeated in 6 jobs because Spark is lazy. Persisting 4.8 MB made evaluation 2.8× faster. Skew was mild (longest task 1.15× the median), so repartitioning's 231 MB shuffle cost more than it saved. No spill was found; two parallel fits gained only 4%.

![Fig. 3.1: before/after times and the task timeline of the longest stage](<task3/figures/task3_optimisation.png>)

### **3.2 Stability Analysis**

| **Model** | **Perturbation applied** | **Δ RMSE (°C)** | **Δ R²** | **Rank** |
|---|---|---|---|---|
| FMRegressor | 10% sensor noise | +0.029 | −0.0017 | 1 |
| RandomForestRegressor | same | +0.072 | −0.0039 | 2 |
| LinearRegression | same | +0.077 | −0.0044 | 3 |
| GBTRegressor | same | +0.086 | −0.0047 | 4 |

GBT is least stable: 80% of its split importance is on today's TMAX, so one wrong reading moves many trees at once (Table 3.2c). Random Forest spreads its splits, so errors partly cancel. FM is stable only because its small weights under-fit. Random Forest keeps the lowest noisy RMSE (3.865 °C), so I would deploy it.

![Fig. 3.2: delta RMSE per perturbation and how far forecasts move](<task3/figures/task3_stability.png>)

### **3.3 Explainability and Fairness**

| **Item** | **Result** |
|---|---|
| Method used | SHAP (exact TreeSHAP on the saved Spark trees) |
| Model explained | GBTRegressor |
| Top five features | 1. TMAX 2. TMAX_roll7 3. doy_cos 4. TOBS 5. PRCP |
| Named bias risk | Extreme-weather bias, worst in Great Plains states |

TMAX carries 66% of SHAP weight; its effect flattens at the extremes (Fig. 3.3c) because trees average training days and records are rare. Days ≥35 °C are forecast 1.9 °C too cold and days ≤−10 °C 5.4 °C too warm. State error follows weather swing (ρ = 0.73), not station count (ρ = 0.17). Mitigation: add pressure and front data and weight extreme days; the limit is that extremes stay rare.

![Fig. 3.3: SHAP importance, top-five beeswarm and the TMAX effect](<task3/figures/task3_shap.png>)

**Task 3 Process Note**

Repartitioning to 24 partitions made GBT retraining 22% slower: skew was only 1.15× and the 231 MB shuffle cost more. I kept 11 partitions and persisted the scored test set.



**Task4 - Tableau Storytelling and Critical Reflection**

\- \- \- \- \- \- \- \- \- \-



Notebook: [`notebooks/task4.ipynb`](notebooks/task4.ipynb) · Tableau data in `task4/tableau/` · results in `task4/results/` and `results/task4.json` · dashboard images in `task4/dashboards/`

**Tableau Workbook (Public Link):** [7006SCN Manish Saini – next-day TMAX forecasting](https://public.tableau.com/app/profile/manish.saini8855/viz/7006SCNManishSaininext-dayTMAXforecasting/D1Dataqualitypipelinemonitoring)

### **4.1 Dashboard Contact Sheet**

![Figs. 4.1–4.4: the four Tableau dashboards](<task4/figures/task4_contact_sheet.png>)

Figs. 4.1–4.4: D1 data quality & pipeline monitoring, D2 model performance & feature importance, D3 business insights, D4 scalability & cost. Each title states the finding; every model keeps one colour on all four dashboards (persistence grey); missing values are a year × feature heat map, error is a state map, and the warning chart plots hit rate against false alarms so the trigger trade-off is visible.

### **4.2 Narrative Insights and Critical Reflection**

| **#** | **Observation** | **Evidence (figure ref)** | **Decision it supports** |
|---|---|---|---|
| 1 | At a 35 °C trigger the models catch 50–58% of hot days (≥ 35 °C); persistence catches 71%. | Fig. 4.3: hit rates; forecasts 1.5–2.0 °C too cold at 35–40 °C | Trigger RF heat alerts at 33 °C: 79% caught, 38% false alarms. |
| 2 | SNOW and SNWD missing rose from 9–12% (train) to 22–23% (test), yet all 8 checks passed. | Fig. 4.1: drift bars | Add a missing-share drift alert; refit the imputer on recent years. |
| 3 | RF is most accurate (3.79 °C) and best under sensor noise, at a cost similar to GBT. | Figs. 4.2, 4.4 | Deploy RF as a nightly batch job; GBT's 0.9 s Task 2 tie-break win was noise. |

Limitation: the 5% station sample leaves half the states with 1–3 test stations (top-ranked Florida has one), so the map compares stations, not states, and 2× day-to-day timing noise means Fig. 4.4 only separates cheap LR from the other three models. I would select models on event hit rates, not RMSE alone. Serving (Day 9) still needs drift monitoring and scheduled retraining, which I have not tested.

**Task 4 Process Note**

RMSE alone failed as the business measure: top-ranked RF missed 48% of ≥35 °C days at a 35 °C trigger. I added hit-rate tables per trigger and recommended 33 °C.
