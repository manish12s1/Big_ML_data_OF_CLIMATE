# 7006SCN2627SEPNOV – Machine Learning and Big Data

## Student Information

- **Student Name:** Manish Saini
- **Student ID:** 17142113
- **Module:** 7006SCN – Machine Learning and Big Data
- **Module Leader:** Dr Katerina Stamou
- **Assessment:** Applied Core Assessment
- **Dataset Pool reference:** 112

## Dataset

- **Dataset:** NOAA Global Historical Climatology Network Daily (GHCN-Daily), US stations, mirrored to Hugging Face as `CatLikeIceCream/temperature`
- **Dataset Source:** Hugging Face
- **Direct URL:** https://huggingface.co/datasets/CatLikeIceCream/temperature

The dataset is one CSV file, `climate.csv`. It holds daily weather records from US weather stations, 1970–2023. The original data comes from NOAA (the US National Oceanic and Atmospheric Administration).

I measured the file in my Task 1 notebook:

| Check | Result |
|---|---|
| Rows | 17,097,453 |
| Columns | 35 |
| File size | 2,730,667,293 bytes (2.54 GiB) |
| Partitions after loading | 21 |

Each row is **one station, one month and one type of measurement**:

- `ID` – station code, for example `USC00010160`
- `YEAR`, `MONTH`
- `ELEM` – measurement type (132 different codes), for example `TMAX` = maximum temperature, `PRCP` = precipitation
- `DAY1` … `DAY31` – one value per day of the month, stored in tenths (`228` = 22.8 °C)

## Notes / Access Caveats

- **Download:** free, no account needed. The notebook downloads the file with `hf_hub_download` only if it is not already in `data/`.
- **Licence:** the Hugging Face mirror does not state a licence. The original NOAA GHCN-Daily data is US-Government public-domain data; NOAA asks users to cite Menne et al. (2012).
- **Not in this repository:** `data/`, `lakehouse/` (Parquet files) and `models/` are too large for GitHub, so `.gitignore` excludes them. The notebook rebuilds them.
- **Memory limits:** I ran the notebook on a Windows laptop with 16 GB RAM. Spark crashed with Java "insufficient memory" errors, so I lowered the driver memory to 4 GB, moved Spark's temporary files to my E: drive, and built the features on a 5% sample of stations.
- **Windows setup:** Spark needs `winutils.exe` and `hadoop.dll` (in `E:\hadoop\bin`, with `HADOOP_HOME` set) to write Parquet files on Windows. The first code cell sets this up.

## GitHub Repository

- **GitHub Repository URL:** https://github.coventry.ac.uk/7006SCN-2627SEPNOV/sainim10_17142113

```
README.md
requirements.txt
.gitignore
notebooks/
    task1.ipynb                 Task 1 notebook (with outputs)
metadata/
    task1_metadata.json         dataset, environment and artefact details
results/
    task1.json                  Task 1 results + handoff (read by Task 2)
    task1_results.json          same content, repository layout name
figures/
    .gitkeep
    T1_1_completeness.png       Fig. T1.1 missing values per day column
```

## Tableau Public

- **Tableau Public URL:**

## Big Data Problem

I want to predict **tomorrow's maximum temperature** at a US weather station from that station's recent weather. This is a **regression** problem, because the answer is a number in °C. Forecasts like this help with energy-demand planning and heat or frost warnings.

The data is hard to handle with ordinary tools:

1. **It is big.** 2.54 GiB and over 17 million rows – too much for one pandas DataFrame on a laptop.
2. **The layout is wide.** One row is a whole month. Turning it into one row per station per day makes the table much longer.
3. **It is not clean.** Days are missing, some values are out of range, and short months have empty day columns.

I used **PySpark** because it splits the file into partitions and processes them in parallel. In my run, Spark used 21 partitions on 12 CPU cores, and a full row count took 5.34 seconds.

## Task 1 – Problem, Data and Pipeline

Notebook: [`notebooks/task1.ipynb`](notebooks/task1.ipynb)

In Task 1, I defined the problem, proved the dataset is Big Data, measured its quality, and built a PySpark pipeline that prepares the data for the models in Task 2. I did not train a prediction model in Task 1.

### 1.1 Problem, dataset and why they fit

- **Problem:** regression – predict next-day maximum temperature (°C) per station.
- **Target:** `label_TMAX_next` = `TMAX` of the next calendar day at the same station. It is only created when the next row really is the next day.
- **Why this dataset fits:** every station is a long daily time series of six useful measurements (`TMAX`, `TMIN`, `TOBS`, `PRCP`, `SNOW`, `SNWD`), so I can build lag, rolling and seasonal features per station.
- **Limitation:** there is no latitude, longitude or elevation. Location is only the US state taken from the station ID. Stations also have gaps, so lag features must check that the earlier day exists.

### 1.2 The five V's, licence and ethics (Table T1.3 in the notebook)

| V | Evidence from my notebook |
|---|---|
| Volume | 17,097,453 rows × 35 columns; 2.54 GiB; 21 partitions |
| Velocity | historical batch (NOAA updates the source daily); full scan in 5.34 s on 12 cores |
| Variety | 132 measurement codes in one column; 31 wide day columns; mixed units |
| Veracity | 14.16–14.60% of `DAY1`–`DAY28` are empty, 50.29% of `DAY31`; 0 duplicate rows; invalid values counted in Table T1.1 |
| Value | tomorrow's `TMAX` correlates r = 0.935 with today's `TMAX` (Table T1.2) |

**Volume** and **Veracity** matter most here: the reshape was too big for my laptop's memory, and gaps or bad values directly damage the lag features and the target.

- **Licence:** public-domain NOAA data; cite Menne et al. (2012).
- **Ethics:** no personal data – the records describe weather stations, not people. The risk is **coverage bias**: some states and decades have fewer stations, so errors may be higher there.

### 1.3 Data preparation and the PySpark Pipeline

The notebook runs in this order:

1. **Load** the CSV with a fixed schema (faster than letting Spark guess, and bad values become `NULL`).
2. **Check** rows, columns, file size and partitions (all above the Big Data minimums).
3. **Profile quality:** missing values (Fig. T1.1), duplicates, and invalid values per measurement (Table T1.1).
4. **Bronze layer:** save the six measurements I use as Parquet, so the CSV is read only once.
5. **Sample:** keep a repeatable 5% of stations (`crc32(ID) % 100 < 5`). Whole stations are kept, so each time series stays complete.
6. **Clean and reshape:** remove duplicates, turn 31 day columns into rows, drop out-of-range values, the `32767` error code and dates like 31 February, then pivot back to **one row per station per day** and convert tenths to normal units.
7. **Features:** yesterday's and the day before's `TMAX` (only if those days exist), 7-day averages, daily temperature range, season as sine/cosine of the day of the year, year, and state. Result: **2,989,300 station-day rows** (silver layer).
8. **Split by time:** train 1970–2015 (2,555,512 rows), test 2016–2023 (433,788 rows). A random split would leak the future into training.
9. **Pipeline** – five stages, fitted on the training years only:

| Order | Stage | Why | Why here |
|---|---|---|---|
| 1 | `Imputer` (median) | fills missing numbers; median because rain and snow are skewed | the assembler fails on empty values |
| 2 | `StringIndexer` | state code → number; `keep` handles unseen states | the encoder needs numbers |
| 3 | `OneHotEncoder` | number → 0/1 vector, no false order between states | needs stage 2 |
| 4 | `VectorAssembler` | joins all features into one `features` vector | needs stages 1 and 3 |
| 5 | `StandardScaler` | puts features on a similar scale; keeps vectors sparse | needs the vector from stage 4 |

The fitted pipeline produces a 54-value feature vector.

### Evidence pack (EP1)

Section 11 of the notebook prints, in one cell: my name, email and dataset, the SparkSession settings, `printSchema()`, the row and column count, the file size, the partition count and the fitted pipeline stages.

### Output and how to check it

- `results/task1.json` – all key numbers, plus a `handoff` block that tells Task 2 where the frozen train/test data and fitted pipeline are.
- `metadata/task1_metadata.json` – dataset, environment and artefact summary.
- `figures/T1_1_completeness.png` – Fig. T1.1.

### What I learned

- Spark is lazy: nothing runs until an action such as `count()`, so an error can appear later than the line that caused it.
- A wide table must be reshaped before modelling; one `stack` + `pivot` is much lighter than many filters and joins.
- For time-series data, both the split and the lag features must respect time, or the model sees the future.
- On a laptop, memory and disk limit Spark as much as the code does. Reading the Java crash report showed me the real cause of my crashes.

## Task 2 – Modelling at Scale

This section will be added when Task 2 is finished.

## Task 3 – Evaluation, Optimisation and Explainability

This section will be added when Task 3 is finished.

## Task 4 – Tableau Storytelling and Critical Reflection

This section will be added when Task 4 is finished.
