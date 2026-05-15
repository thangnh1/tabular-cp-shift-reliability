# Data Sources

All datasets used by this paper are **publicly available** and **downloaded automatically** at runtime by the experiment scripts. No manual download is required. We do **not** redistribute any dataset; the scripts pull each dataset directly from its public host on first use and cache it locally.

| Dataset | Source | License / Terms | Auto-download | Approx. size |
|---------|--------|-----------------|---------------|--------------|
| ACSIncome (4 OOD splits: state, temporal, sex, race) | Hugging Face mirror [`birkhoffg/folktables-acs-income`](https://huggingface.co/datasets/birkhoffg/folktables-acs-income) (a mirror of US Census ACS PUMS preprocessed by the [folktables](https://github.com/zykls/folktables) suite). | Original ACS PUMS data are public-domain U.S. government records. The folktables suite is BSD-licensed. The mirror is a public Hugging Face dataset. | Yes, via `datasets.load_dataset("birkhoffg/folktables-acs-income")`. | ≈ 200 MB cached |
| UCI Adult (Adult sex shift) | OpenML id=2, name `adult` version 2. Originally from UCI Machine Learning Repository, Adult Census Income (Kohavi 1996). | Public-domain U.S. census derivative. | Yes, via `sklearn.datasets.fetch_openml(name="adult", version=2)`. | < 5 MB |
| Bank Marketing (Bank age shift) | OpenML id=1461, name `bank-marketing` version 1. Originally Moro et al. 2014. | Public dataset, OpenML mirror. | Yes, via `sklearn.datasets.fetch_openml(name="bank-marketing", version=1)`. | < 5 MB |
| Taiwan Credit Default (Taiwan sex shift) | OpenML id=42477, `default-of-credit-card-clients`. Originally Yeh & Lien 2009. | Public dataset, OpenML mirror. | Yes, via `sklearn.datasets.fetch_openml(data_id=42477)`. | < 10 MB |
| Diabetes 130-US Hospitals (Diabetes race shift) | OpenML id=4541. Originally Strack et al. 2014. | Public dataset, OpenML mirror. | Yes, via `sklearn.datasets.fetch_openml(data_id=4541)`. | < 20 MB |
| SpeedDating (SpeedDating race shift) | OpenML id=40536. Originally Fisman et al. 2006. | Public dataset, OpenML mirror. | Yes, via `sklearn.datasets.fetch_openml(data_id=40536)`. | < 5 MB |
| Online News Popularity (OnlineNews channel shift) | OpenML id=4545. Originally Fernandes et al. 2015. | Public dataset, OpenML mirror. | Yes, via `sklearn.datasets.fetch_openml(data_id=4545)`. | < 20 MB |

## Cache locations

- Hugging Face datasets cache: `~/.cache/huggingface/`
- scikit-learn / OpenML cache: `~/scikit_learn_data/`

Cumulative cache after a full reproduction: < 1 GB.

## Rejected dataset

| Dataset | Source | Why rejected |
|---------|--------|--------------|
| HELOC (Home Equity Line of Credit) | OpenML id=43890 | On every (CP variant × calibrator × learner × seed) cell we evaluated, this mirror produced a Brier score of exactly 0.000, a textbook signature of label leakage. Excluded from the final 10-task panel. See `results/dataset_quality_notes.md` for the full diagnostic. |

## Known network issue

The `folktables` Python package itself fetches ACS PUMS data directly from `www2.census.gov`, which currently 403s many automated requests behind Cloudflare. Our scripts therefore use the Hugging Face mirror `birkhoffg/folktables-acs-income`, which serves the same upstream PUMS data preprocessed identically to the official folktables suite. If the Census Bureau's PUMS endpoints become reachable again, switching back to the official `folktables` loader is a single-import change in `src/block1_full.py`.

## What this repository does NOT redistribute

We do not include any dataset content in this repository. The scripts download data at runtime from the public hosts listed above; the user's local cache is the only on-disk copy.
