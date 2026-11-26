# ReviewLens project and interview guide

## 1. The project in 30 seconds

ReviewLens is a local command-line tool that classifies English product-review sentences as positive or negative. It uses TF-IDF features and logistic regression, supports single-review and CSV batch predictions, and shows which words or phrases influence the result. It achieved 77.3% accuracy and 0.773 macro-F1 on 198 held-out sentences from the Amazon subset of UCI's Sentiment Labelled Sentences dataset.

This is a small, reproducible classical-ML project. It is not a generative AI agent, an Amazon product, a deployed service, or a model trained on private customer data. The code was prepared with AI assistance. Learn it, run it, and describe the work you actually performed when discussing personal contributions.

## 2. What is implemented

- Offline training using a bundled public dataset; no API key or cloud account required.
- Duplicate and conflicting-label handling before the split.
- A fixed stratified training/test split with seed 42.
- Five-fold model selection on the training set only.
- A saved vectorizer/classifier pipeline and version-pinned dependencies.
- Held-out metrics, all test predictions, and a CSV of misclassified examples.
- Single-input and batch CSV inference, with invalid-input handling.
- Word/phrase contribution explanations for the linear classifier.
- Four automated unit tests. These do not prove production readiness or model quality on new domains.

There is no browser UI, REST API, deployment, live-data feed, latency benchmark, or calibrated confidence system in the current implementation.

## 3. Running it

Use Python 3.12. Open a terminal inside the project folder.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell instead
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python predict.py --text "The battery lasts all day and the sound is excellent."
python predict.py --csv examples.csv --output predictions.csv
python train.py
python -m unittest discover -p "test_*.py" -v
```

The bundled artifact is sufficient for prediction. Training regenerates the artifact and reports. Use the pinned dependencies to reproduce the run. A different library version may change serialization compatibility or results. Never load an untrusted joblib file: it uses pickle-compatible serialization, which can execute code.

Single-review output includes `label`, `positive_score`, `evidence`, and `note`. Batch output adds `model_label`, `model_positive_score`, and `model_note` to the original CSV columns. The input CSV must contain `text`. Empty or over-10,000-character reviews are invalid. All-unknown-vocabulary reviews return `unrecognized` rather than a fabricated sentiment judgment.

## 4. File-by-file map

| File | What to explain |
|---|---|
| train.py | Reads the Amazon subset, cleans records, splits data, searches six configurations, saves the winning pipeline and reports. |
| predict.py | Loads the pipeline, validates text, returns scores and feature contributions, or processes a CSV row by row. |
| test_project.py | Four tests for duplicate/conflict cleaning, split isolation, invalid/unknown input, and logit attribution arithmetic. |
| requirements.txt | Exact package versions used by the project. |
| models/sentiment.joblib | TF-IDF vocabulary, IDF weights, classifier coefficients and intercept in a saved Pipeline. |
| data/source.zip | Unmodified UCI source archive; only the Amazon cells subset is used in training. |
| data/split.csv | Processed text, label, normalized key, and train/test assignment. |
| reports/metrics.json | Final model settings, metrics, confusion matrix, software versions, and archive checksum. |
| reports/cv_results.csv | Training-fold results for every candidate configuration. |
| reports/test_predictions.csv | All 198 held-out predictions and scores. |
| reports/errors.csv | The 45 held-out classification mistakes. |
| examples.csv | Demonstration inputs; not evaluation data. |

## 5. Data and cleaning

The original Amazon subset has 1,000 sentences selected to be clearly positive or negative. Labels are 0 = negative and 1 = positive. This is the 2015 UCI Sentiment Labelled Sentences dataset, not a large current Amazon review corpus.

`clean_data` creates a key by lowercasing and collapsing whitespace. It removes every normalized-text group containing conflicting labels, then retains one row per remaining normalized key. The output has 987 rows. The difference of 13 is the total removed by this process; do not claim all 13 were conflicting labels.

Cleaning occurs before splitting so matching normalized sentences cannot land in both sets. Near-duplicates, paraphrases, related products, and related authors can still cross the split. The dataset does not include product or author IDs for group-based evaluation.

The split contains 789 training sentences and 198 test sentences. Stratification approximately preserves class proportions. A fixed seed makes the random split reproducible; it does not make the sample representative or guarantee generalization.

## 6. TF-IDF, from words to numbers

A classifier needs numeric input. `TfidfVectorizer` tokenizes text into features and returns a sparse matrix. Default preprocessing lowercases words; default word tokens have at least two characters. It does not remove stop words in this project. Punctuation is largely discarded by tokenization.

For a term occurring tf times in a sentence, sublinear term frequency is `1 + ln(tf)` when tf > 0. With default smoothing, inverse document frequency is `ln((1 + N)/(1 + df)) + 1`, where N is the number of training documents and df is how many contain the term. The transformed rows are L2-normalized by default.

Words occurring in fewer training documents get larger IDF values. This does not mean they are automatically useful: a rare typo can also get a high weight. The classifier must learn predictive weights from labels.

Unigrams are individual tokens. Bigrams are adjacent two-token phrases, such as `not good`. They can capture some local negation but do not understand arbitrary sentence structure. The best selected configuration uses both unigrams and bigrams.

The matrix is sparse: each review uses only a small portion of the vocabulary, so only nonzero entries need storage. Avoid converting the entire corpus to a dense array when scaling up.

## 7. Logistic regression

The model computes a linear score `z = b + sum(w_j * x_j)`, then applies the sigmoid `p = 1 / (1 + exp(-z))`. A score p >= 0.5 is labeled positive by the inference function.

Training minimizes logistic classification loss with regularization. `C` is inverse regularization strength: smaller C penalizes large weights more strongly; larger C permits a closer fit to training patterns. The search compares C = 0.5, 1, and 4. The selected value is 4; that is best among the tested candidates on these training folds, not a universally optimal value.

`max_iter=1000` is the optimizer's iteration cap. `random_state=42` is supplied, but the fixed data/CV splits are the central reproducibility controls. The project uses scikit-learn's default logistic-regression solver, not a handwritten gradient-descent implementation.

## 8. Model selection and leakage

The grid contains two n-gram choices times three C values: six configurations. Five-fold cross-validation fits each configuration five times, yielding 30 fold fits, then refits the winner once on all 789 training sentences. Selection maximizes mean validation macro-F1.

The vectorizer lives inside the Pipeline. Each fold therefore learns its own vocabulary and IDF from the fold's training portion. Fitting TF-IDF on all rows before cross-validation would leak validation-corpus information into the representation.

The 198 test sentences are excluded from fitting and configuration selection. They provide the final reported estimate. Reading test errors is useful for diagnosis, but once those errors guide improvements, use fresh validation/test data for the next honest evaluation.

## 9. Results you should know

| Metric | Result |
|---|---:|
| Training CV macro-F1 | 0.820 |
| Test accuracy | 77.3% |
| Test macro-F1 | 0.773 |
| Majority baseline accuracy | 50.5% |
| Majority baseline macro-F1 | 0.336 |
| Correct / total test predictions | 153 / 198 |
| Incorrect test predictions | 45 |
| Accuracy improvement | 26.8 percentage points |

Confusion matrix, with actual labels in rows and predicted labels in columns:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 75 | 25 |
| Actual positive | 20 | 78 |

Taking positive sentiment as the positive class: precision = 78/(78+25) = 75.7%; recall = 78/(78+20) = 79.6%; F1 = 2PR/(P+R) = 77.6%. Macro-F1 averages negative-class F1 (76.9%) and positive-class F1 (77.6%) equally. It is not the same as positive-class F1.

Accuracy = (75+78)/198 = 77.27%. The majority baseline predicts negative for every input and gets 100/198 = 50.51%. The improvement is percentage points, not a 26.8% relative improvement. A small test set produces an uncertain estimate; it does not establish production accuracy.

## 10. Explanations and failure analysis

For each active feature, the explanation computes `TF-IDF value * coefficient`. Positive contributions increase positive-class log-odds; negative contributions decrease them. The six largest absolute contributions are displayed. The intercept and all omitted contributions still affect the score. These are model explanations, not causal explanations or a guarantee of correct sentiment.

The raw logistic score is not calibrated confidence. A score of 0.9 does not establish that nine of ten similar predictions will be correct. Calibration would require a separate evaluation and a calibration procedure fitted without using the final test set.

Actual examples in errors.csv include `Incredible!.`, labeled positive but predicted negative at roughly 0.490, and a long complaint beginning `For a product that costs as much as this one does`, labeled negative but predicted positive at roughly 0.612. These illustrate sparse vocabulary/context limitations. Do not claim a precise cause until inspecting each example's features and contributions.

The sample CSV includes mixed sentiment and unknown vocabulary. A binary model still forces recognized mixed reviews into one of two sentiment labels. `Unrecognized` is an input-vocabulary fallback, not a learned neutral class.

## 11. Likely interview questions with sample answers

### Why did you choose this project?

“I wanted an end-to-end ML project where I could show the full path from labeled text to reproducible evaluation and usable inference. Product reviews give a concrete classification task, and the small dataset let me focus on leakage prevention, baseline comparison, and explaining predictions.” Use this only if it reflects your motivation.

### Why logistic regression instead of a transformer?

“A sparse linear baseline is quick to train and easy to inspect on a small dataset. I did not benchmark transformers, so I cannot claim it outperforms them. I would justify a larger model with validation results and deployment requirements.”

### What is the difference between training and inference?

“Training learns vocabulary, IDF values, and model weights from labeled examples. Inference freezes those parameters and transforms new text into scores. New user reviews do not automatically update the model.”

### How did you prevent leakage?

“I removed matching normalized texts before splitting and placed TF-IDF inside the cross-validation Pipeline. The final holdout was excluded from model selection. This addresses exact-duplicate and vocabulary leakage, but not every possible near-duplicate or product-level correlation.”

### Why use cross-validation if you already have a test set?

“Cross-validation chooses hyperparameters within training data. The test set estimates the selected model's performance independently of that choice. Using test accuracy to pick C would compromise that independence.”

### Why was CV performance higher than test performance?

“They use different samples, and selecting the highest CV result can introduce optimism. The final test macro-F1 was 0.773 versus 0.820 in CV. I report the lower held-out result rather than using the selection score as final performance.”

### How would class imbalance change your approach?

“I would inspect class-specific precision/recall, compare against a majority baseline, and consider class weights or threshold selection on validation data. I would choose metrics around business costs and avoid relying on accuracy alone.”

### What if missing negative reviews is more expensive?

“I would treat negative sentiment as the operational target and measure negative recall and precision. I could move the positive-score threshold upward so more cases are routed as negative, choosing it on validation data. The tradeoff is more false negative-sentiment alerts.”

### What does the probability mean?

“It is the logistic model's positive-class score. I have not established calibration, so I do not describe it as verified confidence. For dependable probabilities I would evaluate calibration and fit an appropriate method using training/validation data.”

### What did your tests actually verify?

“They verify normalized duplicate/conflict handling, no exact-key overlap across train/test, rejection of blank/oversized text and fallback for unknown vocabulary, and the arithmetic behind a short example's attribution. They do not prove semantic correctness, generalization, or production reliability.”

### How does the explanation work?

“Logistic regression is linear in TF-IDF features. Each feature contributes x_j times w_j to log-odds. I display the six largest absolute contributions; omitted terms and the intercept remain part of the full prediction.”

### How would you scale batch processing?

“The current CSV command loops over rows and reads the full file into memory. I would process CSV chunks and call vectorized transform/predict_proba on batches. I would measure throughput and peak memory before reporting a scaling improvement.”

### What is the inference complexity?

“Tokenization depends on input length. For the linear decision step, work is roughly proportional to the number of nonzero features per input. The current explanation sorts all active contributions, about O(k log k) for k active features; a top-six heap could reduce that selection cost. This is algorithmic reasoning, not a measured latency result.”

### How would you deploy it?

“I would wrap trusted model loading and input validation in an API, load the model once per worker, version the artifact, add request limits and monitoring, and benchmark before choosing infrastructure. An AWS container or managed endpoint is a possible extension; this version has not been deployed.”

### What would you monitor after deployment?

“Input length, unknown-vocabulary rates, label distribution, latency, and failure rates. With newly labeled samples, I would track accuracy, class metrics, and calibration. Distribution drift can flag a problem, but labeled evaluation is needed to confirm performance drift.”

### What is your biggest limitation?

“The dataset has fewer than a thousand unique labeled examples and deliberately excludes neutral sentiment. Random splitting does not establish performance on new products or recent reviews. I would prioritize a fresh representative dataset before model complexity.”

### What would you improve first?

“I would collect a new labeled set with mixed and neutral reviews, define whether the task should be binary or multiclass, and benchmark a simple alternative such as Naive Bayes using training-only validation. I would reserve a fresh final holdout.”

### What was your personal contribution?

Describe the parts you actually built, modified, debugged, evaluated, or tested. Do not memorize a claim that you implemented everything unaided. A truthful answer can explain how you used AI assistance, what you verified, what design decisions you understand, and the extension you personally completed.

### Tell me about a challenge using STAR.

Use a real event. A preparation template is: Situation — a text-classification evaluation was at risk of duplicate leakage; Task — make the estimate more trustworthy; Action — normalize keys, remove duplicate/conflicting groups, and verify split isolation; Result — reproducible evaluation on 987 retained rows with a disjoint 198-row test set. Claim this as your own story only if you actually performed and understood those steps. Do not invent a before/after accuracy or a production incident.

### How does this relate to Amazon?

“Review analysis is an understandable customer-feedback use case, and the project demonstrates Python, ML fundamentals, evaluation, and software testing. I do not claim Amazon affiliation, internal data access, or that a sentiment classifier alone establishes readiness for the role.”

## 12. Prepare before submitting

1. Run inference on positive, negative, mixed, blank, and unknown inputs.
2. Explain the difference between accuracy, precision, recall, and macro-F1 using the actual confusion matrix.
3. Trace train.py and predict.py without reading a prepared script aloud.
4. Inspect several errors and distinguish an observed failure from a guessed explanation.
5. Run all four tests and explain what each does not cover.
6. Make one small change yourself, such as chunked batch inference or a new validation test, and document it.
7. Keep resume claims tied to measured results; do not claim deployment, deep learning, production usage, or calibrated confidence.

## 13. Resume bullets

- Built a TF-IDF and logistic-regression sentiment classifier, achieving 77.3% accuracy and 0.773 macro-F1 on 198 held-out product-review sentences.
- Removed duplicate/conflicting records before a stratified split and selected model settings through five-fold cross-validation, improving test accuracy by 26.8 percentage points over a majority baseline.
- Implemented batch CSV inference, feature-contribution explanations, and four automated tests covering cleaning, split isolation, input handling, and attribution arithmetic.

## 14. Data credit

Kotzias, D. (2015). Sentiment Labelled Sentences. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C57604 . Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . The source archive is unmodified; derived CSVs deduplicate and annotate the Amazon subset. See README.md for full attribution and source URL.

All numerical results in this guide come from reports/metrics.json and reports/test_predictions.csv in this project. No external performance benchmark is implied.
