# Project Log — Humor Intelligence

An engineering narrative of how this project was built: the decisions, the
dead ends, the bugs, and how each was diagnosed and fixed. Written to document
the process, not just the result, because the debugging is where most of the
real learning happened.

---

## 1. Goal and scope

Build an NLP project on computational humor using only free compute (local CPU,
Google Colab, Kaggle). The chosen framing: a model that predicts how
funny a joke is, benchmarked against a classical baseline.

## 2. Dataset choice

Selected **rJokes** (Weller & Seppi, LREC 2020) — ~432k Reddit r/Jokes posts
with humor-score labels — over smaller humor datasets because:
- It's large (satisfying the "work with a real dataset" goal).
- It's a published benchmark, so results are comparable to prior work.
- It runs on free compute.

### Key early insight: the label is already log-scaled

Initial EDA assumed the `score` column was the raw Reddit score. Inspecting the
raw source data revealed the labels are actually `round(ln(raw_score + 1))` i.e.
integers 0-11, while raw scores run into the millions. The EDA notebook had been
log-transforming an already-log-transformed value (a double transform).
Fixed by using the label directly as the regression target. It
changed the modeling approach.

## 3. Data cleaning

Manual inspection of top/bottom-scoring jokes surfaced data-quality issues that
a naive pipeline would miss. Quantified and handled:

- **Exact duplicates:** 5,707 in train (1.65%) Reddit reposts. Removed.
- **Ultra-short / title-only entries:** ~0.2% at a <5-word threshold (junk like
  "Classic", "Title", emoji-only). Removed. (A higher threshold was rejected
  because it would delete legitimate one-line puns.)
- **Cross-split leakage:** ~2.4% of dev/test jokes were exact copies of training
  jokes. This would have inflated evaluation on memorized examples. Removed all
  dev/test jokes appearing in train.

Cleaned split sizes: **339,499 train / 41,941 dev / 41,957 test.** Class balance
(bucketed) was checked before and after cleaning and barely shifted, confirming
cleaning removed junk without distorting the target.

## 4. Baseline: TF-IDF + Ridge

Built a classical baseline to establish a reference number the transformer must
beat. An initial run with a **capped vocabulary** (20k features) scored Spearman
**0.302**. Testing vocabulary size showed the cap was costing real accuracy:

| Vocabulary | Test Spearman |
|---|---|
| capped 20k | 0.302 |
| capped 100k | 0.346 |
| uncapped (min_df=5) | **0.363** |

Memory stayed ~150 MB even uncapped (sparse matrices), so the cap was
unnecessary. Final baseline: **Spearman 0.363**, with `min_df=5` to drop the
noisy long tail of rare n-grams.

## 5. Fine-tuning DistilBERT (128 tokens)

Fine-tuned DistilBERT with a regression head (`num_labels=1`,
`problem_type="regression"`) on Google Colab's free T4 GPU.

### Bugs hit and fixed

- **`RuntimeError: Found dtype Long but expected Float`.** Regression loss (MSE)
  needs float labels, but integer scores were being cast back to `Long`. The
  `.map(lambda: float(...))` pattern didn't stick. Fixed with
  `dataset.cast_column("labels", Value("float32"))`, which forces the type at
  the schema level.
- **The DistilBERT "MISSING / UNEXPECTED" load report.** Expected and correct:
  the pretraining head is discarded (UNEXPECTED), the new regression head is
  initialized fresh (MISSING). This is normal for any fine-tune.

### Result

**Test Spearman 0.4038** (Pearson 0.4397), beating the 0.363 baseline. Training
took ~30-45 min. Model pushed to the Hugging Face Hub.

## 6. The 256-token experiment

Hypothesis: a longer context window (256 vs 128 tokens) might help, since ~11%
of jokes are truncated at 128. Ran it as a controlled experiment and change only
`max_sequence_length`, hold everything else constant.

### Infrastructure detours

- **Google Drive ran out of space** (checkpoints). Solved by dropping Drive
  entirely for a ~1h run, checkpoint to local disk and push to HF at the end.
- **Colab free GPU quota exhausted.** Switched to **Kaggle** (separate ~30h/week
  free GPU pool), using **Save & Run All (Commit)** to run server-side so
  disconnects and laptop sleep no longer mattered.

### The confound: multi-GPU silently changed the effective batch size

The first 256 run scored **0.3973** which was worse than 128, with training loss
roughly double (5.4 vs 2.7). Diagnosing from the logs revealed the cause:

- The run finished 2 epochs in **10,610 steps**, but the 128 run took **21,220**
  for the same 2 epochs i.e. half the steps.
- Kaggle provided 2 GPUs (T4 x2). Hugging Face's Trainer splits each batch
  across both, so `per_device_batch=32` became an effective batch of 64.
- Bigger batches = fewer steps per epoch = half the gradient updates = an
  undertrained model. That explained both the higher loss and lower score.

The tell was in the arithmetic: `339,499 / 64 ≈ 5,305 steps/epoch × 2 = 10,610`.
The paired `gather along dimension 0... scalars` warnings confirmed DataParallel
across two GPUs.

This meant the comparison was invalid as the two models differed in *two*
ways (context length AND update count), not one.

### The fix

Set `per_device_train_batch_size = 16` on the 2-GPU setup, giving effective
batch **32** matching the 128 run's **21,220 gradient updates** exactly. Now
only `max_sequence_length` differed. Rerun via Commit mode (~4.5h, unattended).

### Corrected result

**Test Spearman 0.4059** (Pearson 0.4465) — up from the undertrained 0.3973, and
now essentially tied with the 128 model's 0.4038.

## 7. Final results and conclusion

| Model | Context | Test Spearman | Test Pearson |
|---|---|---|---|
| TF-IDF + Ridge baseline | — | 0.363 | — |
| DistilBERT-128 | 128 tokens | 0.4038 | 0.4397 |
| DistilBERT-256 | 256 tokens | 0.4059 | 0.4465 |

**Conclusion:** the fine-tuned transformer clearly beats the classical baseline
(~0.40 vs 0.363). Between 128 and 256 tokens, the difference (0.0021) is within
run-to-run noise so they are statistically equivalent. Since most jokes are
short (median ~18 words), longer context adds little. 128 is preferred for
being ~3x faster to train at no meaningful cost to quality.


## 8. Lessons learned

- **Look at your data by hand.** Duplicates, leakage, and title-only junk were
  found by eyeballing examples, not by metrics.
- **Verify the label means what you think.** The already-log-scaled label would
  have quietly distorted everything.
- **Effective batch size = per-device batch × number of GPUs.** A multi-GPU
  environment silently changed it and invalidated a comparison. Always check the
  step count matches expectations.
- **Free-tier compute is workable with the right workflow.** Kaggle Commit mode
  (server-side, unattended) solved the disconnect problem that plagued Colab.

## 9. Status and next steps

- [x] Data pipeline, EDA, cleaning
- [x] TF-IDF baseline (0.363)
- [x] DistilBERT-128 (0.4038) and DistilBERT-256 (0.4059), both on the Hub
- [ ] Results notebook
- [ ] LLM explanation layer
- [ ] Interactive demo + final writeup
