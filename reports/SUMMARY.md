# Humor Intelligence

## Summary

I built a model that predicts how funny a joke is, trained on 430k+ Reddit jokes
rated by community score. A fine-tuned DistilBERT reaches a test Spearman
correlation of **0.40** against human-derived scores, clearly beating a
TF-IDF + Ridge baseline (0.36). Beyond the headline number, the more interesting
outcome is the analysis. The model ranks jokes reasonably but is poorly
calibrated at the extremes, and a chunk of its errors are actually noise in the
Reddit labels rather than model failures. The whole project runs on free compute.

## The problem

Computational humor is a hard, subjective niche in NLP. What's funny depends on
audience, timing, and context. Rather than trying to generate humor, I framed a
tractable, measurable task, predict a joke's humor score, using a published
benchmark so results are comparable to prior work.

## Data

The **rJokes** dataset (Weller & Seppi, LREC 2020) provides ~432k Reddit r/Jokes
posts, each with a humor label. An early and important finding: the label is not
the raw Reddit score but `round(ln(raw_score + 1))`, a log-compressed integer
from 0 to 11. Treating it correctly (as an already-log-scaled regression target,
not re-transforming it) shaped the rest of the modeling.

Careful cleaning mattered more than expected. Inspecting examples by hand
surfaced three issues: 5,707 exact-duplicate reposts,
thousands of title-only fragments with no actual joke, and
~2.4% of dev/test jokes that were exact copies of training jokes. That last one
is evaluation leakage. Without removing it, the model would be scored partly on
examples it had memorized. After cleaning: 339k train / 42k dev / 42k test.

## Approach

I built up in three stages so each result has a reference point:

A **TF-IDF + Ridge baseline** established the number to beat. Tuning it revealed
that an uncapped vocabulary (with a minimum document frequency to drop rare
noise) reached Spearman **0.363** which was a meaningful jump over a naively capped
version, at negligible memory cost thanks to sparse matrices.

A **fine-tuned DistilBERT** with a regression head was the main model, trained on
free Colab/Kaggle GPUs. It reached test Spearman **0.4038**, beating the
baseline.

A **controlled sequence-length experiment** tested whether a longer context
(256 vs 128 tokens) helps, since ~11% of jokes are truncated at 128 tokens.

## Results

| Model | Context | Test Spearman | Test Pearson |
|---|---|---|---|
| TF-IDF + Ridge baseline | — | 0.363 | — |
| DistilBERT-128 | 128 tokens | 0.4038 | 0.4397 |
| DistilBERT-256 | 256 tokens | 0.4059 | 0.4465 |

The transformer clearly beats the classical baseline. The 128-vs-256 difference
(0.0021) is within run-to-run noise. They are statistically equivalent. Because
most jokes are short (median ~18 words), the longer context adds little, so the
128 model is preferred for being cheaper to train at no quality cost. Both models
reproduce their training scores exactly when reloaded from the Hub (0.4038 and
0.4059), and their prediction behavior is near-identical.

## What the model actually learned

The analysis is more revealing than the score. Both models were evaluated the
same way and behave near-identically; the figures below are the 128 model's,
with the 256 model's equivalents in `reports/figures/` (`*_256.png`). Two
figures tell the story:

The **predicted-vs-true plot** shows the model's mean prediction rising with the
true score but the line is far flatter than the ideal diagonal.
The model compresses nearly all predictions into the 1–5 range and never
confidently predicts an extreme. This is classic regression-to-the-mean:
trained with mean-squared-error on an imbalanced, noisy target, the model hedges
toward the safe middle because confident extreme predictions are punished
heavily when the noisy label disagrees.

The **confusion matrix** (over four humor buckets) makes the same point sharply:
the model is strong on the common middle classes and essentially never predicts
"not funny (0)" or "very funny (6+)". Macro-F1 is only 0.278, low precisely
because it weights the rare classes equally, exposing what plain accuracy would
hide.

The **error analysis** is the most interesting part. The model's biggest "misses"
fall into two groups. Some are label noise: jokes the model rated funny but that
carry a label of 0, including one whose own text reads "Reddit Gold and Front
Page!" suggesting that the label, not the prediction, is wrong. Others are
genuinely hard, short, contextual, culturally specific one-liners that a small
model can't be expected to rate. The takeaway is that the dataset's label noise
is itself part of the performance ceiling. No model can perfectly predict a
target that is partly arbitrary.

## Limitations

Humor is subjective and the labels reflect one Reddit community's tastes, shaped
by timing and virality as much as quality. The model regresses to the mean and
is unreliable at the extremes. The 128-vs-256 result rests on single runs; a
multi-seed study would be needed to make any small difference rigorous. And the
model is a humor *ranker*, not a judge of objective funniness which is appropriate for
research and demos, not high-stakes use.

## Reproducibility

Everything runs on free compute. The repository includes the full pipeline
(data download, cleaning, baseline, fine-tuning notebooks for both Colab and
Kaggle, and results analysis), and both trained models are public on the
Hugging Face Hub. `PROJECT_LOG.md` documents the engineering process in detail, including
the bugs and how they were diagnosed.

## Possible future work

Exploratory directions, not part of the current scope: a multi-seed significance
study of the sequence-length comparison; a larger encoder such as RoBERTa; and an
LLM-based layer that explains a joke's predicted score or cross-checks the model's
humor rankings against an LLM's own judgments. These may or may not be pursued.