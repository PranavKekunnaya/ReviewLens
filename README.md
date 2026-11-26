# ReviewLens

An explainable product-review sentiment classifier built with Python, scikit-learn, TF-IDF, and logistic regression. Classifies English review sentences as positive or negative and shows which words pushed the prediction in either direction. Includes a trained model, offline dataset, batch CSV inference, and reproducible evaluation.

## Quick start

Requires Python 3.12. From this folder:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python predict.py --text "The battery lasts all day and the sound is excellent."
python predict.py --csv examples.csv --output predictions.csv
```

No API keys or paid services required. This is a command-line project, not a hosted application. The bundled model was trained with the versions in requirements.txt; retrain if your scikit-learn version differs. Load only trusted joblib artifacts.

## Reproduce the experiment

```bash
python train.py
python -m unittest discover -p "test_*.py" -v
```

Training reads the bundled UCI source archive. No download is needed. It normalizes case/whitespace to remove exact duplicate sentences and excludes conflicting-label groups before splitting. The fixed, stratified split uses seed 42, with 80% training and 20% testing. The TF-IDF vocabulary is learned within each training fold through a scikit-learn Pipeline. A five-fold search compares unigram versus unigram/bigram features and C values of 0.5, 1, and 4. Selection uses training cross-validation macro-F1; the test set is evaluated only after selection. No manual test-driven tuning was performed.

## Measured results

| Measurement | Result |
|---|---:|
| Original Amazon sentences | 1,000 |
| Unique, nonconflicting sentences | 987 |
| Training / test sentences | 789 / 198 |
| Best training CV macro-F1 | 0.820 |
| Held-out accuracy | 77.3% |
| Held-out macro-F1 | 0.773 |
| Majority baseline accuracy | 50.5% |
| Majority baseline macro-F1 | 0.336 |

Best model: word unigrams/bigrams, sublinear TF-IDF, logistic regression C=4. Confusion matrix (rows actual, columns predicted; negative then positive): [[75, 25], [20, 78]]. Test accuracy is 26.8 percentage points above baseline. These are results on a small public benchmark, not live Amazon traffic or production performance.

See reports/metrics.json for full metrics and environment versions, cv_results.csv for model selection, test_predictions.csv for every held-out prediction, and errors.csv for the 45 mistakes. data/split.csv records the exact split.

## How it works

1. TF-IDF gives each word/phrase a weight based on its frequency in a sentence and relative rarity in the training corpus.
2. Logistic regression learns how those features associate with positive and negative labels.
3. A score of at least 0.5 maps to positive. It is a model score, not calibrated confidence.
4. Explanations show the six largest absolute TF-IDF × coefficient contributions. Positive values increase the positive-class log-odds. Contributions describe model behavior, not causal reasons or factual truth; unshown terms and the intercept also affect the result.

Unknown-vocabulary inputs return `unrecognized`; blank or oversized inputs are rejected. Batch scoring preserves invalid rows with an explanatory status. The model does not have a trained neutral class.

## Project structure

- train.py: dataset cleaning, fixed split, cross-validation, fitting, reports.
- predict.py: single-text and batch inference plus feature attribution.
- test_project.py: duplicate/conflict handling, train/test isolation, input validation, explanation arithmetic.
- models/sentiment.joblib: trained pipeline.
- data/source.zip: original UCI archive; training uses only amazon_cells_labelled.txt.
- data/split.csv: processed Amazon sentences and split assignments.
- reports/: measured evaluation and error analysis files.
- examples.csv: illustrative demo inputs, not evaluation data.

## Limitations and next steps

The 2015 dataset deliberately selects clearly positive/negative sentences and is nearly balanced. It is small and focused on cell-phone/accessory reviews, so it is not representative of all products, languages, neutral reviews, or current customer sentiment. Random splitting does not assess temporal or product-level generalization; the data lacks product/user IDs. Exact duplicates are removed, but semantic near-duplicates may remain. Sarcasm and long mixed-sentiment reviews can fail. Scores have not been calibrated. This is a classical ML baseline, not a deep-learning system or an Amazon-affiliated product.

Before making this your application project, run it yourself, inspect errors, and make a change you can explain. A useful extension is a separate labeled dataset from another product category. Compare models using training-only validation, then evaluate once on a new untouched test set. Avoid repeatedly tuning against this already-reported test set.

## Dataset credit

Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. https://doi.org/10.24432/C57604

Original paper: Dimitrios Kotzias, Misha Denil, Nando de Freitas, and Padhraic Smyth, *From Group to Individual Labels using Deep Features*, KDD 2015.

Dataset page: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

Original archive: https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip

Data license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ . The original archive is unchanged. Derived split/prediction CSVs normalize/deduplicate or annotate its Amazon subset. Source SHA-256 is recorded in reports/metrics.json. Original data authors do not endorse this project. Code is MIT licensed; data remains under CC BY 4.0.
