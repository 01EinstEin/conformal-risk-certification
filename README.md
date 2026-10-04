# Invoice Risk Guard

**A small, auditable risk controller for AI-extracted invoice fields.**

Small teams increasingly have access to OCR and language models, but a confidence score alone does not tell them how often an automated result is wrong. Reviewing every field is expensive; accepting every high-scoring result can silently pass errors through accounting workflows.

Invoice Risk Guard is a research prototype for the decision between **auto-accept** and **human review**. It does not extract invoice fields itself. It accepts confidence scores from an existing extractor and correctness labels from a human audit, then estimates whether a fixed confidence threshold can meet a chosen per-field error-risk budget.

## Research question

> With a small, field-level audit set, can a team automatically accept a useful fraction of AI-extracted invoice fields while keeping the conditional error risk below a predeclared limit—and reliably suspend automation when the confidence-score distribution changes?

The first prototype studies this with synthetic records. It makes no claim that the method is already novel or validated for a real business. A literature review, instructor feedback, and user interviews are part of the research, not afterthoughts.

## What is implemented

- **Field-level risk calibration:** scans a predeclared grid of confidence thresholds and selects the highest-coverage threshold whose one-sided exact binomial upper confidence bound is below the configured risk limit.
- **Multiple-comparison control:** uses a Bonferroni adjustment across tested thresholds; multi-field calibration also adjusts across fields.
- **Safe abstention:** if no threshold meets the risk limit, the policy routes every field to human review.
- **Confidence-shift alarm:** compares reference and recent confidence-score distributions with a two-sample empirical CDF distance and a Dvoretzky–Kiefer–Wolfowitz bound.
- **No external dependencies:** the prototype uses Python's standard library.

The demo's 5% risk budget is illustrative, not a recommended accounting policy.

## Quick start

From this directory:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 experiments/synthetic_demo.py
```

## Mathematical outline

For a fixed threshold \(t\), an audited field is accepted when its reported confidence is at least \(t\). If \(n_t\) audited fields pass and \(k_t\) are wrong, the prototype computes a one-sided Clopper–Pearson upper confidence bound on the conditional error probability. It tests only the thresholds supplied in advance and applies a Bonferroni correction so choosing the highest-coverage passing threshold does not reuse an uncorrected confidence level.

For drift, it measures the largest difference between the reference and recent empirical CDFs of confidence scores. A conservative two-sample DKW bound flags sufficiently large differences. A flagged change disables auto-acceptance through the decision helper.

## Important limits

- Risk bounds rely on the audit examples and future examples being independent or exchangeable within the same operating distribution. A change in supplier mix, scanner quality, model, or workflow can invalidate that assumption.
- The drift check sees **confidence-score distribution changes only**. It can miss a change where scores look the same but their relationship to correctness has changed. Random human audits remain necessary.
- Small audit samples may certify no useful threshold. In that case, this prototype intentionally recommends human review rather than inventing certainty.
- Synthetic experiments are not evidence of production performance. Never upload real invoices, account details, or customer documents to this repository.
- This project does not yet parse documents, call an AI model, or connect to accounting software.

## Repository map

```text
src/invoice_risk_guard/   Data types, risk calibration, drift detection, decisions
tests/                    Standard-library unit tests
experiments/              Deterministic synthetic demonstration
docs/                     Research question, assumptions, and evaluation plan
```

## Research conduct

Before making a novelty claim, compare the method with existing invoice extraction, selective prediction, risk-controlling prediction, and drift-detection work. Any later use of business data must be opt-in, minimized, de-identified where possible, and excluded from public commits.