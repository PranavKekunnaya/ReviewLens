# ReviewLens resume and interview notes

Use these bullets after running, understanding, and verifying the project yourself. This implementation was prepared with AI assistance; describe your own contributions accurately and follow any application disclosure requirements.

**ReviewLens — Product Review Sentiment Classifier | Python, scikit-learn, TF-IDF**

- Built a TF-IDF and logistic-regression pipeline for product-review sentiment classification, achieving 77.3% accuracy and 0.773 macro-F1 on 198 held-out Amazon review sentences.
- Removed duplicate/conflicting review records before a stratified split and selected features and regularization through five-fold cross-validation, outperforming a majority-class baseline by 26.8 percentage points in test accuracy.
- Implemented batch CSV inference, word-level prediction explanations, and four automated tests covering data isolation, duplicate handling, input validation, and attribution correctness.

## Be ready to explain

**Why this model?** The dataset is small; a sparse linear classifier is quick to train, easy to reproduce, and directly interpretable. There is no evidence here that a larger neural model would generalize better.

**Why use a Pipeline?** If vocabulary and inverse document frequencies were learned before cross-validation, validation information could leak into the features. Fitting the vectorizer inside each fold avoids that.

**What does C mean?** It controls inverse regularization strength. Larger C weakens regularization. We tested three values using training-only cross-validation.

**Why report macro-F1?** It averages the class-specific F1 values equally and complements accuracy. It is not the same as positive-class F1.

**What did the baseline do?** It always predicted the most common training label. The test set has 100 negative and 98 positive reviews, so that baseline scored 100/198 = 50.5%.

**Why does CV beat test performance?** They are estimates from different samples; selecting the best CV result introduces selection optimism. The held-out result is the final reported estimate, and a 198-sentence test set has substantial uncertainty.

**What are explanations?** Each feature's TF-IDF value multiplied by its learned coefficient contributes to log-odds. They explain the linear model, not the review's true meaning. Correlated phrases and common words can get surprising weights.

**What would you improve?** Inspect the 45 test errors to describe limitations, collect a fresh external test set, add neutral/mixed labels, and assess category-level generalization. Use new validation data for changes rather than optimize against the existing test results.

## Quick walkthrough

Run a positive, negative, mixed, and unfamiliar-vocabulary example. Inspect reports/errors.csv. Trace one example from text to features to weighted score. Run the four tests. Then implement one extension you understand before presenting the work as your project.
