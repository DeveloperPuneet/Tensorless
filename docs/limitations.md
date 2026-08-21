# Limitations

Tensorless optimizes for "zero setup, works out of the box" over
maximal model quality or feature coverage. Being upfront about the
current limits:

## Models

- The default text tokenizer is **character-level**, not subword/BPE.
  This means no vocabulary file dependency and it works on any language
  out of the box, but it's less token-efficient than production
  tokenizers (GPT-style BPE, SentencePiece), so very long documents take
  more sequence positions to represent, and generation quality per
  parameter is lower than a comparably-sized BPE-tokenized model.
- Models are trained **from scratch** every time — there's no
  fine-tuning of pretrained checkpoints. This keeps things dependency-
  free and fast to set up, but means text-generation quality on small
  datasets is limited by what a small transformer can learn from
  scratch in a short training run, not by the ceiling of large
  pretrained language models.
- Auto-configuration is a **heuristic based on dataset size**, not a
  hyperparameter search. It won't find the optimal architecture for
  your specific data — it finds a reasonable, fast-to-train one.

## Data

- Tabular preprocessing (`TabularPreprocessor`) is fit on the **full
  dataset before the train/val split**, not on the training split alone.
  For small-to-medium datasets this is a common simplification, but it
  is a (typically minor) form of information leakage between train and
  validation statistics.
- No built-in handling for **datetime columns** — they'll currently be
  treated as either numeric (if parseable as a float, which most
  datetime strings aren't) or categorical (with one distinct value per
  unique timestamp, which is rarely useful).
- No built-in handling for **very high-cardinality categorical
  columns** (e.g. free-text IDs) beyond treating every unique value as
  its own category, which produces large, sparse embeddings.
- **Large datasets**: everything currently loads into memory at once
  (`Dataset.texts` / `Dataset.records` are plain Python lists). There's
  no streaming/out-of-core loading yet, so datasets that don't fit in
  RAM aren't supported.

## Format

- The `.tl` format's forward-compatibility story is minimal today:
  Tensorless refuses to load files from a newer format version, but
  there's no migration path yet for loading *older* format versions
  with a newer Tensorless install if the format ever needs a breaking
  change. In practice this means keeping the Tensorless version used to
  load a `.tl` file at or above the version that created it.

## Training

- **Single-device training only** — no multi-GPU or distributed
  training support yet.
- **No mixed dataset formats** — a directory mixing plain text files
  with structured (JSON/CSV) files raises an error rather than trying
  to combine them.
- Early stopping tracks a **single validation metric** (loss); there's
  no support for multi-metric or custom early-stopping criteria yet.

## Scope

Tensorless is not trying to be a general-purpose deep learning research
framework — there's no equivalent of a fully general `nn.Module`
authoring experience for arbitrary architectures, no distributed
training orchestration, and no experiment-tracking integration built in.
It's aimed at "I have a dataset and want a working model with minimum
ceremony," and the models it produces are appropriately sized for that
use case rather than for pushing state-of-the-art benchmarks.

See [roadmap.md](roadmap.md) for what's planned to address some of
these.
