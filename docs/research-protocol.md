# Initial research protocol

## Research question

Can a vendor-agnostic, field-level review gate use a small set of human-audited AI invoice predictions to maximize auto-acceptance coverage while keeping the conditional field error risk below a predeclared limit? How reliably can a separate confidence-score drift alarm detect changes that should suspend automation?

This is a question to investigate, not a novelty claim. Existing invoice extraction and selective prediction research must be mapped before making claims about what is new.

## Mathematical setup

For one field, let each audited prediction be \((S_i, E_i)\), where \(S_i \in [0,1]\) is the extractor's score and \(E_i \in \{0,1\}\) indicates whether the extracted value is wrong according to a human audit. For a fixed threshold \(t\), a prediction is auto-accepted when \(S_i \ge t\). The quantity of interest is

\[
R(t) = \Pr(E = 1 \mid S \ge t).
\]

The calibration module computes a one-sided exact binomial upper bound for \(R(t)\). It tests a predeclared threshold grid and divides the family-wise error probability across all thresholds (and across fields when using `calibrate_by_field`). Among thresholds whose upper bound is no greater than the stated risk limit, it selects the one with the greatest empirical coverage. If none qualifies, it abstains on all records.

For a reference confidence sample and a recent confidence sample, the drift module computes the two-sample empirical CDF distance. A conservative sum of DKW bounds flags a change at the selected significance level. The test concerns the **marginal score distribution**, not the conditional relation between scores and correctness.

## Assumptions and failure modes

- Human labels are correct, consistently defined, and available for each audited field.
- Calibration and future examples are independent or exchangeable within the operating period in which the policy is used.
- The threshold grid is specified before examining audit outcomes.
- A supplier mix, model, OCR quality, language, or workflow change may invalidate the calibration assumption.
- A change in error behavior can occur without a detectable change in the marginal confidence-score distribution. Random audits are required to look for this failure.
- Confidence bounds can be too wide for useful automation when audit samples are small. Abstaining is an intended result, not a bug to hide.

## Evaluation plan

1. **Unit checks:** verify exact zero-error bounds, monotonicity across error counts, abstention on tiny samples, field-specific policies, and drift behavior on tied/discrete scores.
2. **Synthetic stable-distribution experiments:** vary audit-set size, error rate, score calibration, target risk, and number of candidate thresholds. Report auto-acceptance coverage and independent holdout error separately from certification.
3. **Shift experiments:** test confidence-distribution shifts, and separately construct a hard case where the score distribution stays fixed but correctness degrades. The latter should demonstrate the known limit of a score-only alarm.
4. **Baselines:** fixed score thresholds, review-all, and a hindsight oracle reported only as an upper-bound comparator. Use separate calibration and evaluation samples.
5. **Human workflow:** interview small accounting/admin teams about which fields have different error costs, how many fields they review, and what risk budget would be meaningful. Do not choose a business risk limit on their behalf.
6. **Related work:** review invoice OCR/LLM extraction, selective classification, risk-controlling prediction, multiple testing, and two-sample drift detection before describing any contribution as novel.

## Data and privacy

The initial benchmark must be synthetic or public and must not contain personal or business data. If a later pilot uses real documents, obtain explicit permission, minimize retained information, separate raw documents from the repository, and never commit invoices or extracted personal information to GitHub.

## Initial references

- Clopper, C. J. and Pearson, E. S. (1934). “The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.” *Biometrika*, 26(4), 404–413.
- Dvoretzky, A., Kiefer, J., and Wolfowitz, J. (1956). “Asymptotic Minimax Character of the Sample Distribution Function and of the Classical Multinomial Estimator.” *The Annals of Mathematical Statistics*, 27(3), 642–669.
- Massart, P. (1990). “The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality.” *The Annals of Probability*, 18(3), 1269–1283.