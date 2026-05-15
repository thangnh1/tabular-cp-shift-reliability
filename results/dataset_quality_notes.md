# Dataset Quality Notes — 3 Expansion Tasks (v2)

This note documents the three new tabular OOD tasks added in v2 (Diabetes 130-US, SpeedDating, OnlineNews) — their label semantics, OOD split definitions, leakage-prevention steps, and known limitations. Written so a third party can reproduce the analysis from raw OpenML.

---

## Task 1 — Diabetes 130-US Hospitals (OpenML id=4541)

- **Domain**: hospital inpatient records from 130 US hospitals, 1999–2008.
- **Source**: Strack et al. 2014, *BioMed Research International*; OpenML mirror of UCI dataset.
- **n**: 101,766 records.
- **Target**: "readmitted" — 3-class {NO, >30, <30}. We binarize as `<30 = 1` (early readmission), all else = 0. P(y=1) = 0.111.
- **Defensible OOD split**: **race** (5 levels). ID = Caucasian (n=76,099, P(y=1)=0.113); OOD = African American (n=19,210, P(y=1)=0.112). Class rates are nearly equal across ID and OOD → this task primarily tests **covariate shift**, not label-prior shift. Useful as a contrast to Adult/Taiwan/Bank.
- **Feature columns dropped**:
  - `encounter_id`, `patient_nbr`: identifiers that would leak via near-duplicates.
  - `race`: drops the grouping column (otherwise model would trivially memorize the OOD partition).
- **Remaining 46 features**: gender, age (decade buckets), weight (often missing as `'?'`), admission_type_id, discharge_disposition_id, time_in_hospital, num_lab_procedures, num_procedures, num_medications, ..., insulin, change, diabetesMed.
- **Encoding**: categorical strings → integer codes via `pd.Categorical`; NaN → `-1`.
- **Leakage concerns checked**: no column named "readmit" remains in features. Discharge_disposition_id COULD encode "discharged to hospice" or "expired" which would be perfectly predictive — but the dataset DOES include these encounters, treating expired patients as not-readmitted. We do not filter them; this is consistent with Strack et al.'s original methodology and the dataset is widely benchmarked as-is.
- **Verdict**: ACCEPT. Adds medical domain, covariate-shift-dominant case.

---

## Task 2 — SpeedDating (OpenML id=40536)

- **Domain**: speed-dating events at Columbia University.
- **Source**: Fisman et al. 2006, *QJE*; OpenML preprocessed mirror.
- **n**: 8,378 paired interactions.
- **Target**: `match` — 0/1 (did this interaction lead to a match?). P(y=1) = 0.165.
- **Defensible OOD split**: **race of self-identified subject** (5 levels). ID = European/Caucasian-American (n=4,727, P(y=1)=0.167); OOD = Asian/Pacific Islander/Asian-American (n=1,982, P(y=1)=0.135). Class rates differ by 3 pp → **label-prior shift component is present**.
- **Feature columns dropped**:
  - `race`, `race_o` (partner's race), `samerace`, `d_race`, `d_race_o`: anything that directly reveals the race of either participant in the pair.
- **Remaining 115 features**: gender, age, age_o, partner's preferences, evaluation scores on attractive/sincere/intelligent/fun/ambitious/shared_interests, interest in particular activities, etc.
- **Leakage concerns checked**: with race-derived columns removed, leakage risk is on indirect proxies like "interest in particular ethnic cuisines" — we do not filter these (they are legitimate features and not strict proxies).
- **Sample-size caveat**: the Asian OOD population (n=1,982) is the second-largest race subgroup but the calibration set after the 50/25/25 split is ~1,180 — adequate but smaller than the 4,727-row ID group. Weighted-CP density-ratio estimation may suffer somewhat.
- **Verdict**: ACCEPT. Adds social-science domain with race subgroup shift + ~3 pp class-rate shift.

---

## Task 3 — Online News Popularity (OpenML id=4545)

- **Domain**: web articles from Mashable.com, Jan 2013–Jan 2015.
- **Source**: Fernandes, Vinagre, & Cortez 2015, *Progress in AI*.
- **n**: 39,644 articles.
- **Target**: number of shares (continuous). We **binarize at the median (1,400 shares)** → `y = 1` if shares > 1,400. P(y=1) = 0.493 (balanced).
- **Defensible OOD split**: **data_channel** (6 one-hot columns: lifestyle, entertainment, bus, socmed, tech, world). ID = articles with `data_channel_is_tech == 1` (n=7,346); OOD = articles with `data_channel_is_entertainment == 1` (n=7,057). These two channels are similar-sized but topically very different.
- **Feature columns dropped**:
  - All `data_channel_is_*` (6 columns): the grouping variable.
  - `url`: identifier.
  - `timedelta`: days since the article was published — would correlate weakly with cumulative shares.
- **Remaining 53 features**: word counts, title polarity, sentiment scores, presence of images/videos, day-of-week one-hots, average keyword and reference statistics.
- **Leakage concerns checked**: the `kw_*` (keyword statistics) features are computed using shares of other articles in the same period; this is mild data-leakage **risk** (cross-article shares) but it is part of the standard benchmark and we keep it. Same train/cal/test distribution per channel, so within-channel leakage is symmetric.
- **Balanced binary**: median split gives near-50/50 class balance; this is the easiest OOD task of the three for the model.
- **Verdict**: ACCEPT. Adds digital-marketing domain with topic/channel subgroup shift.

---

## Rejected candidates

| Dataset (OpenML id) | Why rejected |
|---------------------|--------------|
| phoneme (1489) | No demographic / topical / temporal column to define a defensible OOD split; only 5 acoustic features. |
| kc1 software defects (1067) | Only 2,109 rows; no plausible shift axis. |
| MagicTelescope (44125) | No demographic features; pure physics measurements with no defensible OOD axis. |

## Why we did *not* include COMPAS

We considered COMPAS-two-years (OpenML id=42193) but rejected it for this v2 because (a) the OpenML mirror is already one-hot encoded for race and sex, so race-based OOD splits would require dropping multiple one-hot columns and the resulting feature set is small, (b) the small sample size (n=5,278) leaves limited room for calibration after the 50/25/25 split, and (c) the dataset has well-known measurement/labelling concerns documented by Larson et al. (2016) that would require dedicated treatment beyond our scope. We may re-evaluate COMPAS for a future expansion with careful re-engineering of the feature set.
