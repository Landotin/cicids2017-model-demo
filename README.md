# CICIDS2017 traffic window demonstration

**[Open the live demonstration](https://cicids2017-model-demo.streamlit.app/)**

Use saved Random Forest models to classify prepared network-traffic windows. Choose **T1** (historical model with a completed final test) or **C+T1 / CT1** (selected study model).

| Evidence | T1 | C+T1 |
|---|---:|---:|
| Mean grouped CV Macro F1 | 71.43% | 72.30% |
| Validation Macro F1 | 77.75% | 78.03% |
| Original test Macro F1 | 85.77% (historical) | 87.31% (previously examined test) |

CT1 test accuracy is **99.52%** and benign false alarms are **2,213 / 417,513 (0.53%)**. This same test was previously examined for T1; CT1 was not fitted on it and was frozen before its own scoring, but this is **not fresh independent confirmation**. No tuning used these scores. The small CV gain does not establish a reliable improvement. See [reviewed results](RESULTS.md) and [prediction reconstruction](REVIEW.json). This app predicts only; it does not train or monitor live network traffic.

## Run on your computer

Use Python 3.12. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Open the local address printed by Streamlit. Both reviewed model files and 31 real validation examples are already included. No Google Drive account is needed.

## Try the demonstration

1. Keep **Reviewed example** to inspect the 31 selected rows, or download the measurements and upload them through **Upload a prepared CSV**.
2. Select T1 or C+T1. For C+T1, optionally show the four calculated inputs.
3. Click **Predict**. Inspect actual labels, predicted labels and mistakes.
4. Download the prediction table. Download example inputs, actual labels and reviewed predictions from the example-files section.

The input has fifteen named numerical measurements for each prepared 60-second conversation window. The app calculates C's extra four inputs itself. It validates counts and totals, rejects malformed files, and never uses labels or identifiers as predictors. The blank template is available in upload mode. A PCAP file cannot be uploaded directly; packet decoding and window preparation happen earlier in the project.

The examples cover fourteen held-out classes and include correct and incorrect predictions. They are selected illustrations, not a benchmark. Heartbleed has no independent held-out examples. No IP addresses or conversation identifiers are published.

## Streamlit Community Cloud setup

After signing in with the intended account, create an app using this public repository, branch `main`, entrypoint `demo/app.py`, and **Python 3.12** under Advanced settings. Dependencies are pinned beside the app. There are no secrets or Drive mounts to configure. After deployment, check both models, predictions, downloads and phone layout before recording a live URL.

The hosted app is live. Both hosted prediction downloads match all 31 saved example predictions, and the example measurements download matches the published bytes. The local fallback above remains available for presentations.

## Reproducibility

`demo/bundle/manifest.json` records both model hashes, library versions, ordered inputs and sample hashes. Startup verifies those files and reproduces every shipped sample prediction. `src/ct1_features.py` is the exact preparation module used during fitting. `demo/bundle/samples/SAMPLE_DATA.md` records selection and provenance. Only trusted bundled model files are loaded; visitors upload CSV measurements.

Public explanations: [project reviewer](https://landotin.github.io/cicids2017-project-reviewer/). Original research repository: [CICIDS2017-Attack-Classification](https://github.com/xtian413/CICIDS2017-Attack-Classification) (project access may be required).

Reviewed 26 October 2026. CT1's test metrics and full row/matrix review are in TEST_EVIDENCE. T1's historical result is unchanged. The separate official-flow comparison is outside the current study/report.
