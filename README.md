# cancer-dx

A Gaussian Naive Bayes classifier that ranks candidate cancer types/stages
from three tumor marker levels: **HE4**, **AFP**, and **CA19-9**.

> **Educational project, not a diagnostic tool.** The class means/variances
> are illustrative values used for a coursework exercise, not values fit to
> a real clinical dataset. This code should never be used to inform an
> actual medical decision.

## Quick start

No install required — the classifier only uses the Python standard library.

git clone https://github.com/<your-username>/CancerDiagnosis.git
cd CancerDiagnosis
python3 run.py

That runs the bundled example patients. Try a single patient of your own:

python3 run.py --he4 180 --afp 6 --ca19-9 22 --name Alice

## Web demo

There's also a browser-based form backed by a small JSON API, with no
framework dependency (built on Python's stdlib `http.server`):

python3 run_web.py

Then open http://127.0.0.1:8000 and enter marker values. Under the hood
it's a `POST /api/predict` endpoint that takes `{"HE4": ..., "AFP": ...,
"CA19-9": ..., "name": ...}` and returns the predicted class plus the
full ranked list as JSON — the same `predict_class()` used by the CLI.

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

run.py            # zero-install CLI entry point
run_web.py        # zero-install web demo launcher
src/cancer_dx/
  model.py          # core Naive Bayes logic (logpdf, class definitions, ranking)
  data.py           # loading patient records from JSON
  cli.py            # command-line interface
  web/
    server.py         # stdlib HTTP server + /api/predict JSON endpoint
    static/
      index.html        # single-page form (no build step)
data/
  example_patients.json
tests/
  test_model.py

## Full install

For the `cancer-dx` console command, the test suite, or using
`cancer_dx` as a library elsewhere:

pip install -e ".[dev]"

Classify the bundled example patients:

cancer-dx

Classify patients from your own JSON file (same shape as
`data/example_patients.json`):

cancer-dx --patients-file path/to/patients.json

Classify a single patient directly from the command line:

cancer-dx --name "Alice" --he4 180 --afp 6 --ca19-9 22

Show more or fewer ranked classes per patient (default is 5):

cancer-dx --top 3

## Running tests

pytest

## Using it as a library

from cancer_dx import predict_class

predicted, ranked = predict_class({"HE4": 180.0, "AFP": 6.0, "CA19-9": 22.0})
print(predicted)          # "Ovarian_Early"
print(ranked[:3])         # top 3 (class, probability) pairs