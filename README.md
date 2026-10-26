# CICIDS2017 traffic window demonstration

**[Open the live demonstration](https://cicids2017-model-demo.streamlit.app/)**

**Traffic Watch** is a simple demonstration of the selected 19-input model. The homepage focuses on choosing traffic, predicting and reading the result. Research settings and recorded test scores are in the collapsed **About this model & example files** section.

## Two ways to explore

- **Try a prediction:** choose one of four real saved validation examples, including a false alarm, or upload a prepared CSV. The result says whether the model flagged the window and names its predicted traffic class. Download the exact prediction table.
- **Workplace simulation:** replay twelve illustrated office events with real saved-model predictions. Play, pause, advance one window, reset or mark an alert as reviewed. The activity feed shows what a network analyst might see; the optional comparison reveals false alarms and missed attacks.

The simulation uses eleven distinct real validation examples, with one normal example deliberately replayed twice. Device aliases, clock times and arrival order are fictional. It has no network connection and takes no firewall, blocking or other operational action. Marking reviewed affects only the current browser session. This is a selected demonstration, not a performance benchmark.

The exact original model and feature formulas are unchanged. The historical T1 model remains in the bundle for reproducibility; the interface uses the selected C+T1 model automatically.

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

1. Under **Try a prediction**, choose an example and click **Predict traffic**.
2. Read the prediction and compare it with the saved label. **A tricky traffic window** shows a real false alarm, so the demonstration also explains mistakes.
3. To try several windows, select **Upload CSV**. Use **Example CSV** or **Blank CSV template** from its collapsed help section. Uploads need the fifteen baseline measurements; the app calculates four additional values itself. A raw PCAP is prepared in the project notebook first.
4. Open **Workplace simulation** and click **Play simulation**, or use **Next window** to explain the workflow step by step. Pause to review an alert. Returning to the prediction tab pauses the replay.
5. Open **About this model & example files** for measurement definitions, recorded test scores, provenance and sample downloads.

Labels, sample IDs and display-only simulated device names are never model inputs. Uploads retain the original checks: finite measurements, count/byte consistency, allowed columns, unique row IDs, no more than 10,000 rows or 2 MB. Results persist across ordinary reruns but are cleared from view when inputs, comparison labels or the model change.

## Streamlit Community Cloud setup

After signing in with the intended account, create an app using this public repository, branch `main`, entrypoint `demo/app.py`, and **Python 3.12** under Advanced settings. Dependencies are pinned beside the app. There are no secrets or Drive mounts to configure. After deployment, check both models, predictions, downloads and phone layout before recording a live URL.

The hosted app is live. Both hosted prediction downloads match all 31 saved example predictions, and the example measurements download matches the published bytes. The local fallback above remains available for presentations.

## Reproducibility

`demo/bundle/manifest.json` records both model hashes, library versions, ordered inputs and sample hashes. Startup verifies those files and reproduces every shipped sample prediction. `src/ct1_features.py` is the exact preparation module used during fitting. `demo/bundle/samples/SAMPLE_DATA.md` records selection and provenance. Only trusted bundled model files are loaded; visitors upload CSV measurements.

Public explanations: [project reviewer](https://landotin.github.io/cicids2017-project-reviewer/). Original research repository: [CICIDS2017-Attack-Classification](https://github.com/xtian413/CICIDS2017-Attack-Classification) (project access may be required).

Reviewed 26 October 2026. CT1's test metrics and full row/matrix review are in TEST_EVIDENCE. T1's historical result is unchanged. The separate official-flow comparison is outside the current study/report.

## Visual audit and verification

The earlier screen put a warning, comparison prose, eight research metrics, a model selector and an input table before the prediction result. The new screen uses two compact panes, a primary action, a readable decision card and optional detail panels. A small CSS robot and office diagram introduce the replay; neither requires external image services or scripts.

Focused checks cover unchanged saved predictions, label exclusion from inference, input/result identity, stale-result prevention, replay provenance and play/pause/review/reset behavior. Run `python -m unittest discover -s tests -v` after installing the pinned requirements. All data and model evidence under TEST_EVIDENCE and demo/bundle remains the historical research record; this interface redesign does not train or re-evaluate a model.
