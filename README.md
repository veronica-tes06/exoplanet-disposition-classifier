# exoplanet-disposition-classifier

Decision tree classifier with a Tkinter GUI that predicts whether a NASA exoplanet candidate is **confirmed**, still a **candidate**, or a **false positive**.

## Features

- Train a decision tree on real exoplanet data from the NASA Exoplanet Archive
- Choose a random seed for the train/test split (1–5)
- Test the model on 250 held-out planets, with a confusion matrix and accuracy score
- Enter values for a new planet and get a predicted disposition

## Data

`exoplanet.csv` was exported from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu). Rows with missing values are dropped before training (about 1,950 planets remain).

| Column | Meaning |
|---|---|
| `period` | Orbital period (days) |
| `impact` | Impact parameter: how close to the star's centre the planet crosses |
| `duration` | Transit duration (hours) |
| `depth` | Transit depth: how much the star dims (ppm) |
| `prad` | Planetary radius (Earth radii) |
| `teq` | Equilibrium temperature (K) |
| `disposition` | Label: `CONFIRMED`, `CANDIDATE` or `FALSE POSITIVE` |

## Getting started

Requires Python 3.9+ with Tkinter (included with the standard Windows and macOS installers).

```bash
git clone https://github.com/<your-username>/exoplanet-disposition-classifier.git
cd exoplanet-disposition-classifier
pip install -r requirements.txt
python Exoplanet_MLGUI.py
```

## Usage

1. Pick a seed and click **Train Data**.
2. Click **Test Data** to see the confusion matrix (rows = actual, columns = predicted) and the accuracy.
3. Fill in the six fields and click **Make Prediction**.

Example inputs from the dataset:

| period | impact | duration | depth | prad | teq | Actual |
|---|---|---|---|---|---|---|
| 9.488 | 0.146 | 2.958 | 615.8 | 2.26 | 793 | CONFIRMED |
| 1.737 | 1.276 | 2.406 | 8079.2 | 33.46 | 1395 | FALSE POSITIVE |

## Results

Accuracy on the 250-planet test set is roughly 68–71% depending on the seed. `CANDIDATE` is the hardest class: it is the smallest group, and candidates by definition haven't yet been confirmed or ruled out, so they overlap heavily with the other two.

## Project structure

```
Exoplanet_MLGUI.py           # Tkinter GUI
ExoplanetMyDecTreeClass.py   # Data loading, training, testing and prediction
exoplanet.csv                # Dataset
```

## Built with

Python · scikit-learn · pandas · Tkinter
