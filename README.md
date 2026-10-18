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
 This is a REGRESSION problem: predicting next-day maximum temperature (°C) at a US weather station, which supports energy-demand planning and heat warnings. The allocated dataset (Pool ref 112) is NOAA GHCN-Daily, US stations, from Hugging Face (CatLikeIceCream/temperature): 17,097,453 rows, 35 columns, 1970–2023  . Each station is a long daily series, so lagged and seasonal predictors can be built per station without using future values. The target, label_TMAX_next, is tomorrow's TMAX at the same station (Table T1.2). Limitation: there are no coordinates or elevation, and station gaps constrain lag features (Fig. T1.1). |
### **1.2 Big Data Justification and Compliance**
*Complete the table using figures from your own dataset. State the licence and note any ethical concerns (privacy, bias, consent).*

| **Dimension**    | **How your dataset demonstrates this dimension**                                                      | **Figure / table ref** |
|------------------|-------------------------------------------------------------------------------------------------------|------------------------|
| Volume           | 17,097,453 rows; 2.54 GiB; 21 partitions; ~60 M station-day rows after reshaping.                     | Fig. EP1, Table T1.3   |
| Velocity         | Historical batch; full scan in 5.34 s (3.2 M rows/s, 12 cores).                                       | Table T1.3             |
| Variety          | 132 element codes; 31 wide day columns; mixed units.                                                  | Table T1.3             |
| Veracity         | DAY1–28 14.16–14.60% null, DAY31 50.29%; 0 duplicates; ⟦p⟧% invalid values. | Fig. T1.1, Table T1.1  |
| Value            | The target correlates r=0.935 with today's TMAX.                                                      | Table T1.2             |
| **Licence**      | None on the mirror; NOAA data is public domain, cite Menne et al. (2012).                             | Task1.ipynb §1         |
| **Ethical note** | No personal data; uneven station coverage risks higher rural error, checked in Task 3.                | Task1.ipynb §9.9       |

*Which of the five V's dominate for this dataset, and why?*

 Volume and Veracity. Reshaping produced about 60 M rows and exhausted my 16 GB laptop, forcing a 5% station sample ; gaps and sentinels corrupt both lags and target.

### **1.3 Data Preparation and PySpark Pipeline**

*Complete the table. Describe each step, justify why it was applied, and name the taught session it comes from.*

| **Stage order** | **Step**     | **Decision taken and justification**                                                             | **Transformer used** |
|-----------------|--------------|--------------------------------------------------------------------------------------------------|----------------------|
| 1               | Impute       | Median, because PRCP and SNOW are skewed; dropping nulls deletes post-gap days (Day 2 workbook). | Imputer              |
| 2               | Index state  | handleInvalid='keep' protects unseen states (Data Pipelines).                                    | StringIndexer        |
| 3               | Encode state | Avoids false ordering of states (Feature Engineering at Scale).                                  | OneHotEncoder        |
| 4               | Assemble     | Single features vector required by Spark ML (Data Pipelines).                                    | VectorAssembler      |
| 5               | Scale        | Unit variance; withMean=False keeps vectors sparse.                                              | StandardScaler       |

*Why must the stages run in this order, and what would break if two were swapped?*

| VectorAssembler rejects nulls, so imputation comes first; OneHotEncoder needs the StringIndexer index; the scaler needs the assembled vector, so swapping any pair fails. All stages are fitted on 1970–2015 only (2,555,512 rows; test 2016–2023: 433,788), so no test statistics leak. |
| 

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
| Writing bronze Parquet failed on Windows because HADOOP_HOME was unset. I installed winutils.exe and hadoop.dll, set HADOOP_HOME before starting Spark, and the 2.54 GiB CSV is now parsed only once. 



**Task2 - Modelling at Scale**

\- \- \- \- \- \- \- \- \- \-



/\*\* Please add here overview of Your Task2 \*\*/



**Task3 - Evaluation, Optimisation and Explainability**

\- \- \- \- \- \- \- \- \- \-



/\*\* Please add here overview of Your Task3 \*\*/



**Task4 - Tableau Storytelling and Critical Reflection**

\- \- \- \- \- \- \- \- \- \-



/\*\* Please add here overview of Your Task4 \*\*/




