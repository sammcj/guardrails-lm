# PII and secrets detection adaptation

## 1. What this is

A sketch of how the current prompt-injection classifier could be repointed at a different job: flagging text that contains personally identifiable information or credentials. The existing system is a binary ModernBERT-base classifier trained on `leolee99/PIGuard` with labels `0=safe`, `1=unsafe`, served via FastAPI `/v1/classify`. For PII/secrets work the architecture stays the same, the endpoint shape stays the same, but the datasets, label semantics, and calibration target all shift.

Two deployment directions, both a good fit for a sequence-level gate:

- **Input-side**: a user's prompt, a tool output, or a RAG chunk is about to be sent to a third-party LLM. Stop the call if it contains an AWS key, an OpenAI token, a customer email address, or a passport number.
- **Output-side**: an LLM response is about to be shown to a user or logged. Stop the response if it echoes PII from retrieved documents, or regurgitates a secret that leaked into training data.

Same model, same endpoint; the caller decides which direction they are gating.

## 2. Architecture choice: binary gate, not token tagging

Two realistic options:

1. **Binary sequence-level classifier**: same `AutoModelForSequenceClassification` head as today, one forward pass per prompt, verdict is `contains_pii_or_secret` yes/no.
2. **Token classification head swap**: `AutoModelForTokenClassification` with BIO or BILUO tags per token, forward pass returns per-token spans.

Pick (1). It reuses the entire training, calibration, ONNX, and serving pipeline unchanged, keeps latency at single-digit milliseconds, and slots into the existing `Classifier` wrapper. The cost is honest: a binary gate loses span information, so it can't redact, only block or route. That's fine for the role this classifier plays. Pair it with Presidio or a regex ensemble when the downstream job needs redacted text rather than a yes/no.

(2) is a separate future project. A token head needs differently-shaped datasets (BIO-tagged), a different eval harness (span-level F1, not sequence accuracy), and a different endpoint contract (returns `list[Span]`). Worth doing later if redaction becomes a product requirement; out of scope for a first adaptation.

## 3. Label schema: binary, with multi-label flagged as future work

Keep `ID2LABEL = {0: "clean", 1: "contains_pii_or_secret"}`. One head, one threshold, one calibration curve. Matches the existing `/v1/classify` response shape so downstream consumers don't change.

Multi-label (`has_email`, `has_credit_card`, `has_aws_key`, `has_api_token`, ...) is a reasonable next step once the binary gate is working. It would need `problem_type="multi_label_classification"` in `AutoModelForSequenceClassification`, a sigmoid per head, per-class thresholds, and a per-class OOD sweep in `ood.py`. All tractable, none of it blocks the binary version from shipping first.

## 4. Datasets

Verified against the HF API. Licences and sizes as of 2026-04.

**Primary training corpus**

- `ai4privacy/pii-masking-300k`: ~300k rows, "other" licence (ai4privacy custom, check before commercial use), EN/FR/DE/IT/ES/NL. Token-level tags; collapse to sequence-level `1 if any PII token else 0`. Good breadth of PII categories and languages.
- `beki/privy`: 100k-400k rows, MIT, English only, token-classification. Generated from Faker-style templates over Presidio entity types. Clean label structure, permissive licence, good training-side pair with the ai4privacy set.

**Augmentation (train side only)**

- `ai4privacy/pii-masking-65k` and `ai4privacy/pii-masking-43k`: smaller ai4privacy siblings with overlapping schema. Useful as diversity if the 300k set saturates. Both exist, licences not explicitly declared on the card, treat as `other` by default.
- Synthetic secrets generation via regex templates plus Faker: generate examples of AWS access keys, Stripe tokens, GitHub PATs, Slack tokens, JWTs, PEM blocks, `.env` fragments. A small helper (`augment_with_synthetic_secrets`) that yields a few thousand positives per pattern is cheaper than chasing a real-world leaked-secret corpus and avoids the ethical question of training on other people's real credentials.
- **Benign hard negatives** (critical — see section 6): a sample of `codeparrot/github-code-clean` (Apache-2.0, large) filtered to files containing strings that look like keys but are example placeholders (`sk_test_`, `AKIAIOSFODNN7EXAMPLE`, `xxxxxxxx`, `YOUR_API_KEY_HERE`). Without these the model blocks every code sample and OpenAPI spec it sees.

**Held-out OOD benchmarks (never in train)**

Mirror the `DEFAULT_REGISTRY` discipline from injection work.

- `ai4privacy/pii-masking-400k` (~400k rows, "other" licence, multilingual): superset-flavoured, hold it out entirely so it acts as a genuine distribution-shift check against the 300k training corpus. If you train on the 400k set instead, swap roles.
- `Isotonic/pii-masking-200k` (cc-by-nc-4.0, non-commercial, multilingual): independent labelling, held out. Non-commercial licence means it's OK for internal benchmarking but never for training a model you'll ship.
- A benign-only held-out set carved from `codeparrot/github-code-clean` README/docs subset: measures FPR on realistic developer text. Pairs with the NotInject-style benign-vocabulary hard-negative role in the injection work.

**Flagged unavailable** (don't waste time hunting): `bigcode/the-stack-secrets` and `bigcode/secrets-in-code` both return HTTP 401 against the HF API as of this writing, the curated filtered-stack secrets sets don't appear to be public. `bigcode/bigcode-pii-dataset` exists but is `gated: manual` (code PII annotations, requires an access request; worth requesting if you want real-code PII coverage). For secrets detection the practical path is regex-generated synthetic plus TruffleHog/Gitleaks test fixtures if you need realistic positives.

## 5. Adaptation steps

1. Preserve the current injection model. `mv checkpoints checkpoints-v2 && mkdir checkpoints` before training. Keep `checkpoints-v2/best/` so `compare-checkpoints` still runs across tasks, even though the tasks differ, it tells you the binary is healthy.
2. Write `augment_with_ai4privacy`, `augment_with_privy`, `augment_with_synthetic_secrets`, and `augment_with_code_benign` helpers in `data.py`. Each normalises to `(text, label)` with label `1` for positive / `0` for benign, calls `.cast(train_subset.features)` before `concatenate_datasets`, and fails soft with `logger.warning` plus `return ds` on load error. Match the existing `augment_with_notinject` / `augment_with_wildjailbreak` template exactly, licences (ai4privacy gated terms, Isotonic's NC clause) should be documented in the docstring for anyone who swaps the dataset.
3. Change `ID2LABEL` / `LABEL2ID` from `{0: "safe", 1: "unsafe"}` to `{0: "clean", 1: "contains_pii_or_secret"}`. Update `LABELS` re-export in `infer.py`. Grep for string matches on label names before merging.
4. Endpoint decision: keep `/v1/classify` as-is. Input shape is still `{prompt: str}`, output shape is still `{unsafe: bool, probability: float, ...}` semantically reused (`unsafe == contains_pii_or_secret`). A separate deployment ships a different trained checkpoint under the same contract. Callers that want both gates run two servers behind different paths. Only introduce a new endpoint if multi-label lands.
5. Add the held-out PII datasets to `DEFAULT_REGISTRY` in `ood.py`. Each gets an `expected_label` for homogeneous sets, a `label_column` if mixed. The code benign set goes in as `expected_label=0`.
6. Calibrate with the FPR-budget mode from `calibration.py`, not F1. False positives block legitimate requests (every code review, every log line with an IP address, every email template), that's the expensive failure in a PII gate. Pick `--max-fpr 0.01` or tighter against the code benign set specifically. F1-optimal thresholds tuned on class-balanced training data will over-fire in production.
7. Before declaring done, run `make eval-ood` against the held-out PII sets and the code benign set, diff against a Presidio baseline on the same inputs. If the classifier can't beat a regex-and-NER ensemble on FPR at matched TPR, the classifier isn't adding value, ship Presidio instead and move on.

## 6. False-positive hazards

The single biggest failure mode for a PII/secrets gate is over-firing on developer text. Without deliberate hard negatives in training the classifier will block:

- Example keys in code and docs: `sk_test_4eC39Hq...`, `AKIAIOSFODNN7EXAMPLE`, `ghp_xxxxxxxxxxxxxxxxxxxx`, `123e4567-e89b-12d3-a456-426614174000`. These appear in every Stripe tutorial, AWS README, and OpenAPI spec.
- Placeholder secrets: `YOUR_API_KEY`, `<token>`, `changeme`, `password123` as dummy values.
- Version numbers that pattern-match IP addresses: `10.0.15.2`, `192.168.1.1` in release notes.
- Serial numbers and product IDs that shape-match SSNs or phone numbers.
- Legitimate public data: published author emails in bibliographies, public company phone numbers, public postal addresses.

Two things follow. First, training on pure PII corpora without a code + docs hard-negative stream is the direct analogue of the v2 injection failure (`docs/failure-analysis-v2.md`): the model learns the surface shape of a positive rather than its meaning. Second, threshold calibration must be done against a benign distribution that looks like production traffic, not just against held-out test rows from the same generator as training. The FPR budget in `calibration.py` is the right lever, pair it with `score_benign_ood` (already supports mixing OOD benign into the calibration pool) so the threshold is fit against code and docs, not synthetic PII-free filler.

## 7. What beats this approach for different needs

This classifier is the fast gate tier. It is cheap per call, stateless, and decides in single-digit milliseconds. That is its job and nothing more.

- **Microsoft Presidio**: mature, open source, combines regex + spaCy NER + context-aware validators, emits spans with entity types so you can redact rather than block. Slower per call, heavier to run, but the right choice when the downstream task is "replace the PII with tokens" rather than "decide whether to proceed". A hybrid is reasonable: classifier as the cheap first pass, Presidio on positives only.
- **Regex ensembles**: `detect-secrets` (Yelp), `TruffleHog`, `Gitleaks`. All three are battle-tested on the secrets side, all three produce span-level output, all three have false-positive reduction heuristics a ModernBERT head won't replicate cheaply. Use them as the authoritative secrets detector in CI/CD pipelines and treat the classifier as the runtime gate for unstructured prompts where a regex ensemble would be too noisy.
- **LLM-as-judge for regulatory-grade PII**: when the cost of a miss is regulatory (HIPAA, GDPR Article 9 special-category data) a classifier at any accuracy is the wrong primary control. Pair tier one (this classifier) with tier two (LLM judge or human review) on positives, and accept the latency cost for the compliance gain.

Framed honestly: the classifier's value is latency and cost per call. Any claim that a 570 MB model beats Presidio on recall for edge-case PII, or beats TruffleHog on verified secrets, is wrong. What it gives you is a sub-10 ms yes/no that you can put in front of every LLM call without thinking about it.

## 8. Open questions

- How much of the Presidio / TruffleHog job does a ModernBERT head actually displace in real traffic? Before investing in training, run Presidio on a day of production prompts, sample the positives, and ask whether a classifier could replace it or merely route to it. The answer changes the design.
- Multilingual coverage: ai4privacy's 300k set covers EN/FR/DE/IT/ES/NL. What happens on Japanese, Korean, Arabic prompts? PIGuard was English-heavy and the injection model showed a clear non-English blind spot. Worth a targeted eval before claiming language coverage the training data doesn't support.
- Combined versus separate models: one head for PII+secrets, or two heads behind the same server? One head simplifies deployment and shares benign hard negatives. Two heads let you tune thresholds independently, which matters because the FP cost of a spurious "secret" block (breaks CI) differs from a spurious "PII" block (blocks a customer email).
- Licence fitness for the ai4privacy corpus: the "other" licence on the 300k and 400k sets is an ai4privacy custom licence. Read it before training a model you intend to distribute. Isotonic's cc-by-nc-4.0 blocks commercial training outright, benchmark-only.
- How noisy is the token-to-sequence collapse? Going from token BIO tags to sequence label `1 if any PII token else 0` throws away count and category. A quick ablation on the 200k set (train on collapsed labels, eval against original token F1 after binarising) would tell us whether we're leaving useful signal on the floor.
