# cancer-dx

A Gaussian Naive Bayes classifier that ranks candidate cancer types/stages
from three tumor marker levels: **HE4**, **AFP**, and **CA19-9**.

> **Educational project, not a diagnostic tool.** The class means/variances
> are illustrative values used for a coursework exercise, not values fit to
> a real clinical dataset. This code should never be used to inform an
> actual medical decision.

## Quick start

No install required — the classifier only uses the Python standard library.

```bash
git clone https://github.com/timjs42/CancerDiagnosis.git
cd CancerDiagnosis
python3 run.py
```

That runs the bundled example patients. Try a single patient of your own:

```bash
python3 run.py --he4 180 --afp 6 --ca19-9 22 --name Alice
```

## How it works

Each candidate class has one *signal* marker — its medically relevant
marker — modeled as a Gaussian (mean, variance):

| Cancer            | Classes                                              | Signal marker |
|-------------------|-------------------------------------------------------|---------------|
| Ovarian           | Early, Late                                            | HE4           |
| Liver (HCC)       | Overall, Stage I, Stage II–III, Stage IV               | AFP           |
| Pancreatic        | Overall, Stage I, Stage II–III, Stage IV               | CA19-9        |

For a given class, any marker that *isn't* its signal marker is scored
under a "healthy" Gaussian instead of being ignored. This prevents, for
example, Pancreatic Stage IV (very high CA19-9) from being misclassified
as Ovarian Early just because HE4 happens to be near-normal — the very
elevated CA19-9 looks extremely unlikely under the healthy CA19-9
distribution and pulls the score for Ovarian Early down.

Given a patient's marker values, the classifier computes
`log P(Class) + log P(signal marker | Class) + sum(log P(other marker | Healthy))`
for every class (assuming a uniform prior), then normalizes with
log-sum-exp to get a probability for each class.

## Project layout

\```
src/cancer_dx/
  model.py   # core Naive Bayes logic (logpdf, class definitions, ranking)
  data.py    # loading patient records from JSON
  cli.py     # command-line interface
data/
  example_patients.json
tests/
  test_model.py
\```

## Installation

\```bash
pip install -e ".[dev]"
\```

## Usage

Classify the bundled example patients:

\```bash
cancer-dx
\```

Classify patients from your own JSON file (same shape as
`data/example_patients.json`):

\```bash
cancer-dx --patients-file path/to/patients.json
\```

Classify a single patient directly from the command line:

\```bash
cancer-dx --name "Alice" --he4 180 --afp 6 --ca19-9 22
\```

Show more or fewer ranked classes per patient (default is 5):

\```bash
cancer-dx --top 3
\```

## Running tests

\```bash
pytest
\```

## Using it as a library

\```python
from cancer_dx import predict_class

predicted, ranked = predict_class({"HE4": 180.0, "AFP": 6.0, "CA19-9": 22.0})
print(predicted)          # "Ovarian_Early"
print(ranked[:3])         # top 3 (class, probability) pairs
\```