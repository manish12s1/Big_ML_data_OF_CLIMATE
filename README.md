 ### **7006SCN2627SEPNOV - Machine Learning and Big Data**

### **Student Information**

\- \- \- \- \- \- \- \- \- \-

#### Student Name: Manish Saini

#### Student ID: 17142113

### Module: 7006SCN - Machine Learning and Big Data

#### **Module Leader**: Dr Katerina Stamou

### **Assessment**: Applied Core Assessment

#### **Dataset**: NOAA Global Historical Climatology Network � Daily (GHCN-Daily), US stations, mirrored to Hugging Face as CatLikeIceCream/temperature

#### **Dataset Source**: Hugging Face

#### **Direct URL**: https://huggingface.co/datasets/CatLikeIceCream/temperature

### Notes / Access Caveats

GHCN-Daily US stations, mirrored to Hugging Face as CatLikeIceCream/temperature. 17.1M rows, 35 columns. Target: temperature or precipitation.

#### **GitHub repository URL**:  https://github.coventry.ac.uk/7006SCN-2627SEPNOV/sainim10_17142113.git

#### Tableau Public URL:



**Big Data problem**  

\- \- \- \- \- \- \- \- \- \-



/\*\* overview here your Big Data problem \*\*/



**Task1 - Problem, Data and Pipeline**  

### **1.1 Problem, Dataset and Why This Pairing**
Define the real-world continuous outcome problem. State explicitly that this is a REGRESSION ML problem predicting a continuous target variable. Name the dataset allocated to you by Your Module Leader and state its source and argue why this dataset suits this problem. State your target variable and what it means. For the higher bands, identify a limitation of the dataset for this problem.

This is a regression problem. I predict tomorrow's maximum temperature (°C) at a US weather station, which helps with energy planning and heat warnings. My dataset (Pool ref 112) is NOAA GHCN-Daily for US stations, from Hugging Face (CatLikeIceCream/temperature), with 17,097,453 rows and 35 columns from 1970 to 2023. Each station is a long daily record, so lag and seasonal features need no future days. The target, label_TMAX_next, is the next day's TMAX at the same station. A limitation: there is no location or elevation data, and record gaps limit the lag features (Fig. T1.1).

### **1.2 Big Data Justification and Compliance**
*Complete the table using figures from your own dataset. State the licence and note any ethical concerns (privacy, bias, consent).*

| **Dimension**    | **How your dataset demonstrates this dimension** | **Figure / table ref** |
|------------------|--------------------------------------------------|------------------------|
| Volume           | 17,097,453 rows; 2.54 GiB; 21 partitions after loading. | Fig. EP1, Table T1.3 |
| Velocity         | Historical batch data; a full scan took 5.34 s on 12 cores. | Table T1.3 |
| Variety          | 132 element codes; 31 wide day columns; mixed units. | Table T1.3 |
| Veracity         | DAY1–28 are 14.16–14.60% empty and DAY31 50.29%; 0 duplicates; invalid values counted per element. | Fig. T1.1, Table T1.1 |
| Value            | Tomorrow's TMAX has r = 0.935 with today's TMAX. | Table T1.2 |
| **Licence**      | None on the mirror; NOAA data is public domain, cite Menne et al. (2012). | Task1.ipynb §1 |
| **Ethical note** | No personal data; states with fewer stations may get higher errors. | Task1.ipynb §9.9 |

*Which of the five V's dominate for this dataset, and why?*

Volume and Veracity. Reshaping the full file was too big for my 16 GB laptop, so I kept 5% of stations. Gaps and error codes such as 32767 damage the lags and the target.

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

VectorAssembler fails on nulls, so imputing comes first. The encoder needs the StringIndexer output and the scaler needs the assembled vector, so swapping any two stages breaks the pipeline. All stages are fitted on 1970–2015 only; the 2016–2023 test years are never used.

### **EP1 Evidence Pack**
*Paste one image, produced from a single notebook cell, showing in order: SparkSession initialisation, df.printSchema(), row and column count, file size verification, partition count after ingestion, and the fitted pipeline stage list.*
 ![SparkSession initialisation,](<figures/Screenshot 2026-10-18 001107.png>)
  [PARTITION COUNT](<figures/Screenshot 2026-10-18 001230.png>)
   ![rows count op](<figures/Screenshot 2026-10-18 001209-1.png>) 
 ![file size verfication o/p](<figures/Screenshot 2026-10-18 001222-1.png>)
 ![5 v,s](<figures/Screenshot 2026-10-18 001315.png>) 
  ![Pipeline construction](<figures/Screenshot 2026-10-18 001348.png>) 
   ![EP 1](<figures/Screenshot 2026-10-18 001336-1.png>)
   ![EP1](<figures/Screenshot 2026-10-18 001358.png>) 
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
| Inference time (s) | 1.71 | 10.11 | 3.13 | 3.29 |

All four beat the "tomorrow = today" baseline (RMSE 4.397). I use RMSE because big misses on very hot or cold days matter most. Random Forest is best, but GBT is only 1.6% worse, which is inside my 2% tie rule. GBT predicts three times faster, so I recommend GBT.

### **EP2 Evidence Pack**

![EP2: best parameters and times for all four models, with actual-vs-predicted and residual plots](<task2/figures/task2_ep2_composite.png>)

*Fig. EP2: one loop prints each model's best parameters and times, above actual-vs-predicted and residual plots for the same 5,143 test days. This is regression, so there are no ROC or confusion-matrix panels.*

**Task 2 Process Note**

Random Forest crashed with Java OutOfMemoryError: maxMemoryInMB=512 across 12 tasks and 2 parallel fits exceeded the 4 GB heap. I reproduced it, set 64 MB, and cross-validation completed.



**Task3 - Evaluation, Optimisation and Explainability**

\- \- \- \- \- \- \- \- \- \-



/\*\* Please add here overview of Your Task3 \*\*/



**Task4 - Tableau Storytelling and Critical Reflection**

\- \- \- \- \- \- \- \- \- \-



/\*\* Please add here overview of Your Task4 \*\*/




