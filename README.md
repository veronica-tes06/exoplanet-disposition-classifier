# exoplanet-disposition-classifier

Machine-learning classifier with a Tkinter GUI that predicts whether a NASA Kepler exoplanet candidate is **confirmed**, still a **candidate**, or a **false positive**.

## Features

- Train a gradient boosting model or a decision tree on all 9,564 Kepler Objects of Interest (KOIs)
- Choose a random seed for the train/test split (1–5)
- Test on a held-out 20% of objects (1,913), with a confusion matrix, accuracy and macro F1
- Look up any KOI by name (e.g. `K00752.01`) and see the predicted class, class probabilities and the real answer
- Or type in six values for a new planet and get a prediction

## Data

`data/cumulative.csv` is the Kepler Objects of Interest *cumulative* table from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu). The label is `koi_disposition`:

| Class | Objects |
|---|---|
| `FALSE POSITIVE` | 4,839 |
| `CONFIRMED` | 2,746 |
| `CANDIDATE` | 1,979 |

The model uses 60 inputs: 58 columns from the table, in five groups, plus two derived features.

| Group | Examples | Why it helps |
|---|---|---|
| Core transit values | period, impact, duration, depth, planet radius, Teq | The basic shape of the dip in brightness |
| Signal and fit quality | `koi_model_snr`, `koi_num_transits`, `koi_ror`, `koi_srho`, measurement errors | Weak or badly fitting signals are often noise |
| Host star | temperature, surface gravity, radius, mass, metallicity, brightness | Puts the planet's size and orbit in context |
| Centroid offsets | `koi_fwm_*`, `koi_dicco_*`, `koi_dikco_*` | If the star's image shifts during the transit, the dip is probably from a background star |
| System | `koi_count`, `koi_tce_plnt_num` | Stars with several signals are rarely false positives |

Two features are derived: transit duration divided by period^(1/3) (compared with Kepler's third law), and a flag for grazing transits (impact ≥ 1).

**Columns that are left out on purpose**, because they leak the answer: `koi_pdisposition`, `koi_score` and the `koi_fpflag_*` flags come from NASA's own vetting pipeline, and `kepler_name` is only given to confirmed planets. Including them pushes accuracy to about 95%, but then the model is just copying NASA's decision.

`data/TOI.csv` (TESS) and `data/k2_pandas.csv` (K2) are included but not used for training. They don't have the signal and centroid measurements that matter most. Pooling them with Kepler on the shared columns improved Kepler accuracy by less than 1 percentage point.

## Getting started

Requires Python 3.9+ with Tkinter (included with the standard Windows and macOS installers).

```bash
git clone https://github.com/<your-username>/exoplanet-disposition-classifier.git
cd exoplanet-disposition-classifier
pip install -r requirements.txt
python Exoplanet_MLGUI.py
```

## Usage

1. Pick a model and a seed, then click **Train Data** (gradient boosting takes a few seconds). The output box suggests some KOIs from the test set.
2. Click **Test Data** to see the confusion matrix (rows = actual, columns = predicted), accuracy and macro F1.
3. Type a KOI name and click **Look Up and Predict**. This uses the full model and also fills in the six fields below. Objects from the training set are marked as not a fair test.
4. Or fill in the six fields yourself and click **Make Prediction**. This uses a separate model trained only on those six values, so it is less accurate.

Example inputs:

| period (days) | impact | duration (hrs) | depth (ppm) | radius (R⊕) | Teq (K) | Actual |
|---|---|---|---|---|---|---|
| 9.488 | 0.146 | 2.958 | 615.8 | 2.26 | 793 | CONFIRMED |
| 1.737 | 1.276 | 2.406 | 8079.2 | 33.46 | 1395 | FALSE POSITIVE |

## Results

5-fold cross-validation on all 9,564 KOIs:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Original decision tree, 6 features, ~2,000 rows | ~71% | 0.59 |
| Gradient boosting, 6 core features only | 72.7% | 0.67 |
| Decision tree (max depth 8, min 20 per leaf), all features | 80.8% | 0.77 |
| Random forest, all features | 85.9% | 0.82 |
| **Gradient boosting, all features** | **86.8%** | **0.84** |

Results per class for gradient boosting:

| Class | Precision | Recall |
|---|---|---|
| CONFIRMED | 0.89 | 0.94 |
| FALSE POSITIVE | 0.90 | 0.92 |
| CANDIDATE | 0.73 | 0.65 |

`CANDIDATE` is still the hardest class. A candidate is a signal nobody has confirmed or ruled out yet, so many look exactly like confirmed planets or false positives. A lot of today's candidates will end up in one of the other two classes. The biggest gains came from the centroid-offset features (+5.7 points) and the signal-quality features (+8 points).

## Project structure

```
Exoplanet_MLGUI.py           # Tkinter GUI
ExoplanetMyDecTreeClass.py   # Data loading, features, training, testing and prediction
data/cumulative.csv          # Kepler KOI table (used for training)
data/TOI.csv                 # TESS Objects of Interest (not used)
data/k2_pandas.csv           # K2 planets and candidates (not used)
```

## Built with

Python · scikit-learn · pandas · NumPy · Tkinter
