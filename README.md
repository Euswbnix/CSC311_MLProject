# CSC311 ML Project — Painting Classification from Survey Responses

A machine learning pipeline that predicts which of three famous paintings a
survey respondent is describing, based on their answers to a 15-question
mixed-format questionnaire (Likert scales, multiple-choice, free text, and
messy free-form numeric input).

The three target classes are:

1. *The Persistence of Memory* — Salvador Dalí
2. *The Starry Night* — Vincent van Gogh
3. *The Water Lily Pond* — Claude Monet

Built for the **CSC311 — Introduction to Machine Learning** course
challenge at the University of Toronto.

## Results

| Metric | Value |
|---|---|
| 5-fold stratified CV accuracy | **88.37% ± 0.83%** |
| Hold-out test accuracy (20% split) | **90.24%** |
| Train − CV gap | 3.19% (well-controlled overfitting) |
| Final model | Logistic Regression (L2, C = 0.3) |
| Feature dimension | 302 (22 structured + 280 TF-IDF text) |

The final submission script `pred.py` runs entirely on **pure NumPy +
Pandas** — no scikit-learn at inference time — to comply with the
challenge restrictions.

## Repository layout

```
CSC311_MLProject/
├── data/
│   ├── ml_challenge_dataset.csv     Raw survey responses (1686 rows × 16 cols)
│   └── cleaned_data.csv             Cleaned & feature-engineered dataset
├── data_cleaning.py                 Raw CSV → cleaned_data.csv
├── model_training.py                Trains & cross-validates 3 model families
├── pred.py                          Pure-NumPy inference (submission entry point)
├── model_params.npz                 Exported weights, scaler, TF-IDF vocab
├── training_results.txt             Logged output of a full training run
├── csc311challenge.pdf              Course assignment specification
└── csc311genAI.pdf                  GenAI usage disclosure
```

## Pipeline

### 1. Data cleaning (`data_cleaning.py`)

The raw survey is messy in characteristic ways. The cleaning script handles:

- **Column renaming** — long survey questions → short Python-friendly names.
- **Likert scales** — `"4 - Agree"` → `4`, etc.
- **Outlier clipping** — `num_colours` and `num_objects` clipped to `[0, 20]`
  (a painting can't have 100 prominent colours).
- **`willing_to_pay` cleaning** — by far the dirtiest column. The script
  copes with `"$1,000"`, `"5 000 000$"`, `"100 million"`, `"300 dollars."`,
  `"a"`, `"pancakes,"` etc. via regex extraction with multiplier handling
  (`million`, `billion`) and an upper sanity bound of 10 M.
- **Multi-select columns** — `season`, `room`, `view_with` are
  multi-label-binarized into 14 individual indicator columns.
- **Missing-value imputation** — numeric columns filled with the median;
  free-text columns filled with the empty string.

The cleaned dataset is saved to `data/cleaned_data.csv`.

### 2. Feature engineering (`model_training.py`)

- **Structured features (22 dim)** — 8 numeric columns + 14 binary
  multi-label columns, normalized with `MinMaxScaler`.
- **Text features (280 dim)** — TF-IDF over three free-text columns:
  - `feeling_desc` → 150 features
  - `food` → 50 features
  - `soundtrack` → 80 features
  - English stop-words removed, `min_df = 5` to filter noise.
- **Combined matrix** — 302 dimensions total.

### 3. Model selection

Three model families are compared with **5-fold Stratified CV**:

| Family | Variants tried |
|---|---|
| **Logistic Regression** (L2) | C ∈ {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10} on both `struct` and `combined` |
| **Naive Bayes** | Gaussian NB on structured, Multinomial NB on text/combined with α ∈ {0.1, 0.5, 1, 2} |
| **Decision Tree** | `max_depth` ∈ {3, 5, 7, 10, 15, None} on both feature sets |

The winner is **Logistic Regression with C = 0.3 on the combined 302-dim
feature set**, beating Naive Bayes by ~0.4 % and Decision Trees by ~8 %
on CV accuracy.

### 4. Inference (`pred.py`)

The submission script must run with **NumPy + Pandas only** (no sklearn),
so all preprocessing has been re-implemented from scratch:

- `clean_price()` — re-implements the `willing_to_pay` regex pipeline.
- Likert / season / room / view-with mappings — pure Python dicts and
  substring checks.
- `tfidf_transform()` — manual TF × IDF + L2-normalisation, mirroring
  `sklearn.feature_extraction.text.TfidfVectorizer`.
- `predict()` — `scores = x @ COEF.T + INTERCEPT`, then `argmax`.

All learned parameters are stored in `model_params.npz`:

```
coef                       (3, 302)   logistic regression weights
intercept                  (3,)       per-class biases
classes                    (3,)       painting names
scaler_min, scaler_range   (22,)      MinMaxScaler parameters
fill_medians               (8,)       per-column medians used at train time
tfidf_<col>_vocab_keys
tfidf_<col>_vocab_vals     vocabulary {word: index}
tfidf_<col>_idf            IDF vector for each text column
```

## Top features

The 5 most influential features (mean |coef| across classes) are
unsurprisingly the most "diagnostic" sensory associations:

| Rank | Feature | |coef| mean |
|---:|---|---:|
| 1 | `tfidf_food::salad` | 0.84 |
| 2 | `season_Spring` | 0.79 |
| 3 | `sombre` (Likert) | 0.79 |
| 4 | `tfidf_food::blueberry` | 0.79 |
| 5 | `num_objects` | 0.76 |

Salad / blueberry / spring → Monet's water-lily pond. "Sombre" and
night-related vocabulary → Van Gogh. Few objects + warped time language
→ Dalí. The model is essentially picking up the human sensory
fingerprints of each painting.

## Requirements

- Python ≥ 3.9
- pandas
- numpy
- scikit-learn *(only for `data_cleaning.py` / `model_training.py`;
  `pred.py` does not require it)*

```bash
pip install pandas numpy scikit-learn
```

## How to run

```bash
# 1. Clean the raw survey
python data_cleaning.py

# 2. Train and cross-validate all model families
python model_training.py

# 3. Run inference on a CSV with the same column layout as the training set
python pred.py path/to/test.csv
```

`pred.py` exposes a `predict_all(filename)` function that returns a list
of painting-name strings, matching the interface required by the course
challenge.

## License

Released under the MIT License.
