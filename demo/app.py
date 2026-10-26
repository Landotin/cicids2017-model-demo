"""Small, offline prediction demo for the reviewed T1 and C+T1 bundles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.ct1_features import BASE_FEATURES, ENGINEERED_FEATURES, make_input
except ImportError as exc:  # Keep the missing implementation visible at startup.
    raise RuntimeError(
        "The CT1 feature contract is unavailable. Keep src/ct1_features.py beside this demo."
    ) from exc

try:
    import streamlit as st
except ModuleNotFoundError:  # Pure validation helpers remain usable in small test environments.
    st = None


PINNED_MODEL_VERSIONS = {
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "scipy": "1.16.3",
    "sklearn": "1.7.2",
    "joblib": "1.6.0",
}
PINNED_STREAMLIT_VERSION = "1.66.1"
MANIFEST_SCHEMA_VERSION = 1
MODEL_NAMES = ("T1", "CT1")
EXPECTED_EVIDENCE = {
    "T1": "historical_tested",
    "CT1": "selected_previously_examined_test",
}
TRAINING_ONLY_LABELS = {"Heartbleed"}
MAX_UPLOAD_BYTES = 2_000_000
MAX_UPLOAD_ROWS = 10_000
ALLOWED_EXTRA_COLUMNS = {
    "sample_id",
    "actual_label",
    "label",
    "conversation_id",
    "capture_file",
    "label_status",
    *ENGINEERED_FEATURES,
}

COUNT_FEATURES = (
    "protocol",
    "packet_count",
    "total_bytes",
    "min_packet_bytes",
    "max_packet_bytes",
    "forward_packet_count",
    "backward_packet_count",
    "forward_bytes",
    "backward_bytes",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
)
NONNEGATIVE_FEATURES = (
    "protocol",
    "duration_seconds",
    "packet_count",
    "total_bytes",
    "mean_packet_bytes",
    "min_packet_bytes",
    "max_packet_bytes",
    "forward_packet_count",
    "backward_packet_count",
    "forward_bytes",
    "backward_bytes",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
)

FEATURE_DESCRIPTIONS = {
    "protocol": "Numeric IP protocol value.",
    "duration_seconds": "Duration of the prepared window, in seconds.",
    "packet_count": "Number of retained packets in the window.",
    "total_bytes": "Total retained frame bytes in the window.",
    "mean_packet_bytes": "Mean retained frame length, in bytes.",
    "min_packet_bytes": "Smallest retained frame length, in bytes.",
    "max_packet_bytes": "Largest retained frame length, in bytes.",
    "forward_packet_count": "Packets in the forward direction.",
    "backward_packet_count": "Packets in the backward direction.",
    "forward_bytes": "Frame bytes in the forward direction.",
    "backward_bytes": "Frame bytes in the backward direction.",
    "syn_count": "Packets carrying the SYN flag.",
    "ack_count": "Packets carrying the ACK flag.",
    "fin_count": "Packets carrying the FIN flag.",
    "rst_count": "Packets carrying the RST flag.",
}


class BundleError(ValueError):
    """The trusted model bundle is missing or does not match its manifest."""


class DemoInputError(ValueError):
    """A CSV row is not a valid prepared-window input."""


@dataclass(frozen=True)
class InputBatch:
    baseline: pd.DataFrame
    sample_ids: pd.Series
    actual_labels: pd.Series | None


@dataclass(frozen=True)
class LoadedBundle:
    root: Path
    manifest: dict[str, Any]
    models: dict[str, RandomForestClassifier]
    sample_inputs: pd.DataFrame
    sample_labels: pd.Series
    sample_predictions: pd.DataFrame
    class_labels: tuple[str, ...]


def _actual_model_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_limited_csv(path: Path) -> pd.DataFrame:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BundleError(f"Required sample file is missing: {path.name}.") from exc
    if size > MAX_UPLOAD_BYTES:
        raise BundleError(f"Sample file {path.name} exceeds the {MAX_UPLOAD_BYTES:,}-byte limit.")
    try:
        content = path.read_bytes()
        _check_csv_header(content, path.name)
        table = pd.read_csv(io.BytesIO(content), nrows=MAX_UPLOAD_ROWS + 1)
    except DemoInputError as exc:
        raise BundleError(f"The bundled sample file {path.name} is invalid: {exc}") from exc
    except (UnicodeDecodeError, pd.errors.ParserError, csv.Error) as exc:
        raise BundleError(f"Could not read sample file {path.name} as a UTF-8 CSV.") from exc
    if len(table) > MAX_UPLOAD_ROWS:
        raise BundleError(f"Sample file {path.name} exceeds the {MAX_UPLOAD_ROWS:,}-row limit.")
    return table


def _check_csv_header(content: bytes, filename: str = "uploaded CSV") -> list[str]:
    if not content:
        raise DemoInputError(f"{filename} is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DemoInputError(f"Keep {filename} below {MAX_UPLOAD_BYTES:,} bytes.")
    try:
        decoded = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DemoInputError(f"{filename} must be a readable UTF-8 CSV.") from exc
    cleaned = [name.strip() for name in header]
    if not cleaned or any(not name for name in cleaned):
        raise DemoInputError(f"{filename} has a blank column name.")
    duplicates = sorted({name for name in cleaned if cleaned.count(name) > 1})
    if duplicates:
        raise DemoInputError("Column names must be unique after trimming spaces: "
                             + ", ".join(duplicates) + ".")
    return cleaned


def _row_name(sample_ids: pd.Series | None, index: int) -> str:
    if sample_ids is not None and index < len(sample_ids):
        return f"row {sample_ids.iloc[index]!r}"
    return f"row {index + 1}"


def validate_input_table(table: pd.DataFrame, class_labels: tuple[str, ...] | None = None) -> InputBatch:
    """Validate prepared-window measurements and keep labels out of predictors."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise DemoInputError("The CSV needs at least one prepared-window row.")
    if len(table) > MAX_UPLOAD_ROWS:
        raise DemoInputError(f"Upload no more than {MAX_UPLOAD_ROWS:,} rows at a time.")
    names = [str(name).strip() for name in table.columns]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise DemoInputError("Column names must be nonblank and unique.")
    clean = table.copy()
    clean.columns = names
    missing = [feature for feature in BASE_FEATURES if feature not in clean.columns]
    if missing:
        raise DemoInputError("Missing required baseline measurements: " + ", ".join(missing) + ".")
    unknown = sorted(set(clean.columns) - set(BASE_FEATURES) - ALLOWED_EXTRA_COLUMNS)
    if unknown:
        raise DemoInputError(
            "Unsupported CSV column(s): " + ", ".join(unknown)
            + ". Start from the downloadable 15-measurement template; known labels and metadata are ignored."
        )

    baseline = pd.DataFrame(index=clean.index)
    for feature in BASE_FEATURES:
        converted = pd.to_numeric(clean[feature], errors="coerce")
        invalid = converted.isna() | ~np.isfinite(converted.to_numpy(dtype="float64"))
        if invalid.any():
            row = int(np.flatnonzero(invalid.to_numpy())[0])
            raise DemoInputError(f"{_row_name(None, row)} has a missing or nonnumeric {feature} value.")
        baseline[feature] = converted.astype("float64")
    baseline = baseline.loc[:, list(BASE_FEATURES)]

    sample_id_column = "sample_id" if "sample_id" in clean.columns else None
    if sample_id_column:
        sample_ids = clean[sample_id_column].astype("string")
        if sample_ids.isna().any() or sample_ids.str.strip().eq("").any():
            raise DemoInputError("Every sample_id must contain a value.")
        sample_ids = sample_ids.str.strip().astype(str).reset_index(drop=True)
    else:
        sample_ids = pd.Series([f"row-{index + 1:05d}" for index in range(len(clean))], name="sample_id")
    if sample_ids.duplicated().any():
        raise DemoInputError("sample_id values must be unique within one upload.")

    label_columns = [name for name in ("actual_label", "label") if name in clean.columns]
    if len(label_columns) > 1:
        raise DemoInputError("Provide one optional label column: label or actual_label.")
    actual_labels = None
    if label_columns:
        values = clean[label_columns[0]].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise DemoInputError("The optional actual-label column must be complete.")
        actual_labels = values.str.strip().astype(str).reset_index(drop=True)
        if "UNLABELED" in set(actual_labels):
            raise DemoInputError("UNLABELED rows cannot be scored in this supervised demo.")
        if class_labels is not None:
            unknown = sorted(set(actual_labels) - set(class_labels))
            if unknown:
                raise DemoInputError("Unknown actual label(s): " + ", ".join(unknown) + ".")

    _validate_arithmetic(baseline, sample_ids)
    return InputBatch(baseline=baseline.reset_index(drop=True),
                      sample_ids=sample_ids, actual_labels=actual_labels)


def _validate_arithmetic(baseline: pd.DataFrame, sample_ids: pd.Series | None = None) -> None:
    values = baseline.loc[:, list(BASE_FEATURES)].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise DemoInputError("Every predictor must be a finite number.")

    for feature in COUNT_FEATURES:
        column = baseline[feature].to_numpy(dtype="float64")
        non_integer = ~np.isclose(column, np.rint(column), rtol=0, atol=1e-8)
        if non_integer.any():
            row = int(np.flatnonzero(non_integer)[0])
            raise DemoInputError(f"{_row_name(sample_ids, row)} has a non-integer {feature} count.")

    invalid_protocol = baseline.protocol.to_numpy(dtype="float64") > 255
    if invalid_protocol.any():
        row = int(np.flatnonzero(invalid_protocol)[0])
        raise DemoInputError(f"{_row_name(sample_ids, row)} has a protocol value outside 0–255.")

    for feature in NONNEGATIVE_FEATURES:
        negative = baseline[feature].to_numpy(dtype="float64") < 0
        if negative.any():
            row = int(np.flatnonzero(negative)[0])
            raise DemoInputError(f"{_row_name(sample_ids, row)} has a negative {feature} value.")

    packet_count = baseline.packet_count.to_numpy(dtype="float64")
    if (packet_count <= 0).any():
        row = int(np.flatnonzero(packet_count <= 0)[0])
        raise DemoInputError(f"{_row_name(sample_ids, row)} needs a positive packet_count.")

    packet_sum = (baseline.forward_packet_count + baseline.backward_packet_count).to_numpy()
    packet_mismatch = packet_sum != baseline.packet_count.to_numpy()
    if packet_mismatch.any():
        row = int(np.flatnonzero(packet_mismatch)[0])
        raise DemoInputError(
            f"{_row_name(sample_ids, row)} has inconsistent packet totals; forward plus backward "
            "packets must equal packet_count."
        )

    byte_sum = (baseline.forward_bytes + baseline.backward_bytes).to_numpy()
    byte_mismatch = byte_sum != baseline.total_bytes.to_numpy()
    if byte_mismatch.any():
        row = int(np.flatnonzero(byte_mismatch)[0])
        raise DemoInputError(
            f"{_row_name(sample_ids, row)} has inconsistent byte totals; forward plus backward "
            "bytes must equal total_bytes."
        )

    for feature in ("syn_count", "ack_count", "fin_count", "rst_count"):
        too_many = baseline[feature].to_numpy() > packet_count
        if too_many.any():
            row = int(np.flatnonzero(too_many)[0])
            raise DemoInputError(f"{_row_name(sample_ids, row)} has {feature} greater than packet_count.")

    minimum = baseline.min_packet_bytes.to_numpy(dtype="float64")
    mean = baseline.mean_packet_bytes.to_numpy(dtype="float64")
    maximum = baseline.max_packet_bytes.to_numpy(dtype="float64")
    average = baseline.total_bytes.to_numpy(dtype="float64") / packet_count
    tolerance = np.maximum(1e-6, np.abs(average) * 1e-5)
    bad_size = ((minimum > maximum) | (mean < minimum - tolerance)
                | (mean > maximum + tolerance) | (maximum > baseline.total_bytes.to_numpy())
                | (np.abs(mean - average) > tolerance))
    if bad_size.any():
        row = int(np.flatnonzero(bad_size)[0])
        raise DemoInputError(
            f"{_row_name(sample_ids, row)} has inconsistent packet-size statistics; check "
            "min_packet_bytes, mean_packet_bytes, max_packet_bytes, and total_bytes."
        )


def parse_uploaded_csv(content: bytes, class_labels: tuple[str, ...] | None = None) -> InputBatch:
    """Parse a bounded upload without letting labels or metadata enter model inputs."""
    _check_csv_header(content)
    try:
        table = pd.read_csv(io.BytesIO(content), nrows=MAX_UPLOAD_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DemoInputError("Could not read the upload. Save it as a UTF-8 CSV and try again.") from exc
    return validate_input_table(table, class_labels=class_labels)


def input_template_csv() -> bytes:
    """Return a header-only CSV with exactly the fifteen model-independent inputs."""
    return pd.DataFrame(columns=list(BASE_FEATURES)).to_csv(index=False).encode("utf-8")


def prepare_model_inputs(batch: InputBatch, model_name: str) -> pd.DataFrame:
    if model_name not in MODEL_NAMES:
        raise BundleError(f"Unknown model name: {model_name}.")
    features = make_input(batch.baseline.loc[:, list(BASE_FEATURES)], engineered=(model_name == "CT1"))
    expected = tuple(BASE_FEATURES) + (tuple(ENGINEERED_FEATURES) if model_name == "CT1" else ())
    if tuple(features.columns) != expected:
        raise BundleError(f"The {model_name} feature code returned an unexpected column order.")
    values = features.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise DemoInputError("Calculated model inputs contain a non-finite value.")
    return features


def _safe_bundle_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise BundleError("The manifest contains a model path outside its bundle.")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise BundleError("The manifest contains a model path outside its bundle.")
    if not path.is_file():
        raise BundleError(f"The model file is missing: {relative}.")
    return path


def _check_versions(manifest_versions: Any) -> None:
    if not isinstance(manifest_versions, dict):
        raise BundleError("manifest.json needs a software_versions object.")
    missing = [name for name in PINNED_MODEL_VERSIONS if name not in manifest_versions]
    if missing:
        raise BundleError("The bundle does not record model-library versions: " + ", ".join(missing) + ".")
    current = _actual_model_versions()
    mismatches = {}
    for name, pinned in PINNED_MODEL_VERSIONS.items():
        recorded = str(manifest_versions[name])
        actual = str(current[name])
        if recorded != pinned or actual != recorded:
            mismatches[name] = {"bundle": recorded, "required": pinned, "runtime": actual}
    if mismatches:
        details = ", ".join(
            f"{name} (bundle {item['bundle']}, required {item['required']}, runtime {item['runtime']})"
            for name, item in mismatches.items()
        )
        raise BundleError("Model-library versions are incompatible: " + details + ".")
    recorded_python = manifest_versions.get("python") or manifest_versions.get("python_version")
    if recorded_python:
        expected_minor = ".".join(str(recorded_python).split(".")[:2])
        runtime_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        if expected_minor != runtime_minor:
            raise BundleError(
                f"Python {runtime_minor} is incompatible with the bundle's Python {recorded_python}; "
                f"use Python {expected_minor}."
            )


def _expected_features(model_name: str) -> tuple[str, ...]:
    return tuple(BASE_FEATURES) + (tuple(ENGINEERED_FEATURES) if model_name == "CT1" else ())


def _check_model(name: str, model: Any, record: Any) -> RandomForestClassifier:
    if not isinstance(record, dict):
        raise BundleError(f"The {name} manifest entry must be an object.")
    expected_features = _expected_features(name)
    if record.get("input_features") != list(expected_features):
        raise BundleError(f"The {name} input feature order differs from the approved contract.")
    if record.get("engineered") is not (name == "CT1"):
        raise BundleError(f"The {name} engineered flag differs from the approved contract.")
    if record.get("evidence_status") != EXPECTED_EVIDENCE[name]:
        raise BundleError(f"The {name} evidence status is missing or inconsistent.")
    if not isinstance(record.get("metrics"), dict):
        raise BundleError(f"The {name} manifest entry needs a metrics object.")
    if not isinstance(model, RandomForestClassifier):
        raise BundleError(f"The {name} artifact must be a Random Forest classifier.")
    if list(getattr(model, "feature_names_in_", ())) != list(expected_features):
        raise BundleError(f"The saved {name} model expects a different input feature order.")
    settings = {
        "n_estimators": 80,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 2,
        "criterion": "gini",
        "max_features": "sqrt",
        "min_samples_split": 2,
        "bootstrap": True,
    }
    changed = [field for field, expected in settings.items() if getattr(model, field, None) != expected]
    if changed:
        raise BundleError(f"The saved {name} model has unexpected forest settings: " + ", ".join(changed) + ".")
    labels = tuple(str(label) for label in getattr(model, "classes_", ()))
    if not labels or "UNLABELED" in labels or len(labels) != len(set(labels)):
        raise BundleError(f"The saved {name} model has invalid class labels.")
    return model


def _load_samples(root: Path, models: dict[str, RandomForestClassifier], class_labels: tuple[str, ...]):
    sample_dir = root / "samples"
    if not (sample_dir / "SAMPLE_DATA.md").is_file():
        raise BundleError("The reviewed example provenance file samples/SAMPLE_DATA.md is missing.")
    inputs = _read_limited_csv(sample_dir / "sample_inputs.csv")
    labels = _read_limited_csv(sample_dir / "sample_labels.csv")
    predictions = _read_limited_csv(sample_dir / "sample_predictions.csv")
    if inputs.columns.tolist() != ["sample_id", *BASE_FEATURES]:
        raise BundleError("sample_inputs.csv must contain sample_id followed by the fifteen baseline features.")
    if labels.columns.tolist() != ["sample_id", "actual_label"]:
        raise BundleError("sample_labels.csv must contain exactly sample_id and actual_label.")
    if predictions.columns.tolist() != ["sample_id", "model", "predicted_label"]:
        raise BundleError("sample_predictions.csv must contain exactly sample_id, model, and predicted_label.")
    input_ids = inputs.sample_id.astype(str)
    label_ids = labels.sample_id.astype(str)
    labels = labels.copy()
    labels["sample_id"] = label_ids.to_numpy()
    labels["actual_label"] = labels.actual_label.astype(str)
    predictions = predictions.copy()
    predictions["sample_id"] = predictions.sample_id.astype(str)
    predictions["model"] = predictions.model.astype(str)
    predictions["predicted_label"] = predictions.predicted_label.astype(str)
    if (input_ids.duplicated().any() or label_ids.duplicated().any()
            or set(input_ids) != set(label_ids) or len(input_ids) != len(label_ids)):
        raise BundleError("Sample inputs and labels must have the same unique sample IDs.")
    if not len(inputs):
        raise BundleError("The reviewed example set is empty.")
    if set(labels.actual_label.astype(str)) & TRAINING_ONLY_LABELS:
        raise BundleError("Built-in examples cannot claim held-out Heartbleed labels.")
    unknown = sorted(set(labels.actual_label.astype(str)) - set(class_labels))
    if unknown:
        raise BundleError("Built-in examples contain unknown labels: " + ", ".join(unknown) + ".")

    try:
        batch = validate_input_table(inputs, class_labels=class_labels)
    except DemoInputError as exc:
        raise BundleError(f"The bundled sample inputs are invalid: {exc}") from exc
    expected_rows = {(sample_id, model) for sample_id in input_ids for model in MODEL_NAMES}
    observed_rows = list(zip(predictions.sample_id, predictions.model))
    if len(observed_rows) != len(set(observed_rows)) or set(observed_rows) != expected_rows:
        raise BundleError("Sample predictions must contain one row per sample for both T1 and CT1.")
    prediction_labels = predictions.predicted_label
    if not set(prediction_labels).issubset(class_labels):
        raise BundleError("Sample predictions contain a class outside the saved model classes.")

    ordered_labels = labels.set_index("sample_id").loc[input_ids, "actual_label"].reset_index(drop=True)
    try:
        for name in MODEL_NAMES:
            features = prepare_model_inputs(batch, name)
            reproduced = models[name].predict(features)
            model_predictions = predictions.loc[predictions.model.eq(name)].copy()
            recorded = (model_predictions.set_index("sample_id")
                        .loc[input_ids, "predicted_label"].astype(str).to_numpy())
            if not np.array_equal(np.asarray(reproduced, dtype=str), recorded):
                raise BundleError(f"The {name} predictions for the reviewed examples do not match the saved models.")
    except DemoInputError as exc:
        raise BundleError(f"The bundled sample inputs are invalid: {exc}") from exc
    return (inputs.copy(), ordered_labels.rename("actual_label"), predictions.copy())


def load_bundle(bundle_dir: str | Path) -> LoadedBundle:
    """Load only a hash-matched bundle under the configured trusted directory."""
    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir():
        raise BundleError(
            f"Trusted model bundle not found at {root}. Complete and review the CT1 export, "
            "then set DEMO_BUNDLE_DIR to its Export directory. This demo does not train or invent predictions."
        )
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"Trusted model bundle is incomplete: {manifest_path} is missing.")
    if manifest_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise BundleError("manifest.json is unexpectedly large.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError("manifest.json is unreadable or invalid JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise BundleError(f"Expected model bundle schema version {MANIFEST_SCHEMA_VERSION}.")
    if manifest.get("baseline_features") != list(BASE_FEATURES):
        raise BundleError("manifest.json baseline_features must match the ordered fifteen-feature contract.")
    _check_versions(manifest.get("software_versions"))
    records = manifest.get("models")
    if not isinstance(records, dict) or set(records) != set(MODEL_NAMES):
        raise BundleError("manifest.json must contain exactly the reviewed T1 and CT1 model records.")

    # Public score display is tied to the reviewed test completion record.
    evidence = REPO_ROOT / "TEST_EVIDENCE/complete.json"
    if not evidence.is_file() or _sha256(evidence) != manifest.get("test_evidence_record_sha256"):
        raise BundleError("The reviewed test evidence is missing or changed.")
    measured = json.loads(evidence.read_text(encoding="utf-8"))
    if (measured.get("selected_model", {}).get("sha256") != records["CT1"].get("sha256")
            or measured.get("fresh_independent_test") is not False
            or measured.get("model_refitted") is not False):
        raise BundleError("The displayed test result does not identify this saved model.")
    for shown, saved in (("test_macro_f1", "supported_class_macro_f1"),
                         ("test_accuracy", "accuracy"),
                         ("test_false_alarm_rate", "benign_false_alarm_rate")):
        if records["CT1"]["metrics"].get(shown) != measured["metrics"].get(saved):
            raise BundleError("The displayed score differs from reviewed test evidence.")

    root_resolved = root.resolve()
    models: dict[str, RandomForestClassifier] = {}
    for name in MODEL_NAMES:
        record = records[name]
        if not isinstance(record, dict):
            raise BundleError(f"The {name} manifest entry must be an object.")
        relative = record.get("model_path")
        if not isinstance(relative, str) or relative != f"models/{name}.joblib":
            raise BundleError(f"The {name} model path must be models/{name}.joblib.")
        path = _safe_bundle_file(root_resolved, relative)
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise BundleError(f"The {name} manifest entry needs a SHA-256 model hash.")
        if _sha256(path) != expected_hash.lower():
            raise BundleError(f"The saved {name} model hash does not match manifest.json.")
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise BundleError(f"The saved {name} model could not be loaded in this environment.") from exc
        models[name] = _check_model(name, model, record)

    if list(models["T1"].classes_) != list(models["CT1"].classes_):
        raise BundleError("T1 and CT1 must have the same ordered class contract.")
    class_labels = tuple(str(label) for label in models["T1"].classes_)
    sample_inputs, sample_labels, sample_predictions = _load_samples(root, models, class_labels)
    return LoadedBundle(root, manifest, models, sample_inputs, sample_labels,
                        sample_predictions, class_labels)


def _bundle_path() -> Path:
    configured = os.environ.get("DEMO_BUNDLE_DIR")
    return Path(configured) if configured else Path(__file__).resolve().parent / "bundle"


if st is not None:
    @st.cache_resource(show_spinner="Loading the classifier…")
    def _cached_load_bundle(bundle_path: str, bundle_fingerprint: str) -> LoadedBundle:
        del bundle_fingerprint  # It is a cache key; the loaded files remain authoritative.
        return load_bundle(bundle_path)
else:
    def _cached_load_bundle(bundle_path: str, bundle_fingerprint: str) -> LoadedBundle:
        del bundle_fingerprint
        return load_bundle(bundle_path)


def _bundle_fingerprint(root: Path) -> str:
    """Invalidate the loaded model when any reviewed bundle file changes."""
    relative_paths = [
        "manifest.json",
        *(f"models/{name}.joblib" for name in MODEL_NAMES),
        "samples/sample_inputs.csv",
        "samples/sample_labels.csv",
        "samples/sample_predictions.csv",
        "samples/SAMPLE_DATA.md",
    ]
    entries = []
    for relative in relative_paths:
        path = root / relative
        try:
            resolved = path.resolve()
            digest = _sha256(path) if path.is_file() else "missing"
        except OSError:
            resolved = path
            digest = "unreadable"
        entries.append((relative, str(resolved), digest))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_cached_bundle(bundle_dir: str | Path) -> LoadedBundle:
    root = Path(bundle_dir).expanduser().resolve()
    return _cached_load_bundle(str(root), _bundle_fingerprint(root))


def predict(batch: InputBatch, bundle: LoadedBundle, model_name: str) -> pd.DataFrame:
    if model_name not in MODEL_NAMES:
        raise BundleError(f"Unknown model name: {model_name}.")
    features = prepare_model_inputs(batch, model_name)
    predicted = bundle.models[model_name].predict(features)
    result = pd.DataFrame({"sample_id": batch.sample_ids, "predicted_label": np.asarray(predicted, dtype=str),
                           "model": model_name})
    if batch.actual_labels is not None:
        result.insert(1, "actual_label", batch.actual_labels.to_numpy(dtype=str))
        result["correct"] = result.actual_label.eq(result.predicted_label)
    return result


def _sample_batch(bundle: LoadedBundle) -> InputBatch:
    return validate_input_table(
        bundle.sample_inputs.assign(actual_label=bundle.sample_labels.to_numpy()),
        class_labels=bundle.class_labels,
    )


def _uploaded_batch(uploaded: Any, class_labels: tuple[str, ...]) -> InputBatch:
    return parse_uploaded_csv(uploaded.getvalue(), class_labels=class_labels)


# Interface helpers use the already verified prediction functions above.
from html import escape
from time import monotonic
from demo.presentation import EXAMPLES, SCENES, SELECTED_MODEL, analyst_note, prediction_key, replay_table, title

BOT = '<div class="tw-bot" role="img" aria-label="Friendly classifier robot"><span class="bot-antenna"></span><div class="bot-head"><div class="bot-face"><i class="bot-eye"></i><i class="bot-eye"></i><span class="bot-smile"></span></div></div><span class="bot-feet"></span></div>'
MONITOR = '<div class="tw-monitor" aria-hidden="true"><span class="monitor-screen"><i></i><i></i><i></i></span><span class="monitor-stand"></span></div>'



def _html(content: str) -> None:
    st.html(content)


def _subset_batch(bundle: LoadedBundle, sample_ids) -> InputBatch:
    whole = _sample_batch(bundle)
    positions = {sample_id: index for index, sample_id in enumerate(whole.sample_ids)}
    indices = [positions[sample_id] for sample_id in sample_ids]
    return InputBatch(whole.baseline.iloc[indices].reset_index(drop=True),
                      whole.sample_ids.iloc[indices].reset_index(drop=True),
                      whole.actual_labels.iloc[indices].reset_index(drop=True))


def _window_summary(batch: InputBatch) -> None:
    if len(batch.baseline) == 1:
        row = batch.baseline.iloc[0]
        stats = [(f'{int(row.packet_count):,}', 'packets'),
                 (f'{row.total_bytes / 1024:,.1f} KB', 'captured bytes'),
                 (f'{row.duration_seconds:,.2f} s', 'observed duration')]
    else:
        stats = [(f'{len(batch.baseline):,}', 'traffic windows'),
                 (f'{batch.baseline.packet_count.sum():,.0f}', 'packets in total'),
                 (f'{batch.baseline.total_bytes.sum() / 1024:,.1f} KB', 'captured bytes')]
    _html('<div class="tw-stats">' + ''.join(
        f'<div class="tw-stat"><strong>{escape(value)}</strong><span>{escape(label)}</span></div>'
        for value, label in stats) + '</div>')


def _prediction_card(label: str) -> None:
    flagged = label != 'BENIGN'
    status = 'Needs analyst review' if flagged else 'No alert from the model'
    explanation = ('The model flagged this window as a possible attack. An analyst would check the alert.'
                   if flagged else 'The model classified this window as normal traffic.')
    _html(f'<div class="tw-result {"attack" if flagged else ""}">'
          f'<div class="tw-status"><span class="tw-dot"></span>{status}</div>'
          f'<h3>{escape(title(label))}</h3><p>{explanation}</p></div>')


def _render_prediction(bundle: LoadedBundle) -> None:
    input_column, output_column = st.columns([1, 1], gap='large')
    batch = None
    with input_column:
        with st.container(border=True, key='input-card'):
            _html('<div class="tw-step">01 / Input</div><h2 class="tw-title">Choose some traffic</h2>')
            source = st.radio('Input source', ['Try an example', 'Upload CSV'], horizontal=True,
                              label_visibility='collapsed', key='source_mode')
            if source == 'Try an example':
                sample_id = st.selectbox('Choose an example', list(EXAMPLES),
                    format_func=lambda value: EXAMPLES[value], key='example_id')
                batch = _subset_batch(bundle, [sample_id])
                st.caption('A real traffic example from our saved dataset.')
            else:
                st.caption('Upload prepared conversation-window measurements. One CSV row is one window.')
                uploaded = st.file_uploader('Choose a prepared CSV', type=['csv'], key='measurements_upload',
                    help=f'Up to {MAX_UPLOAD_ROWS:,} rows and 2 MB. Include the fifteen baseline measurements.')
                if uploaded is not None:
                    try:
                        batch = _uploaded_batch(uploaded, bundle.class_labels)
                    except DemoInputError as exc:
                        st.error(str(exc))
                with st.expander('Get a sample file or template'):
                    st.download_button('Example CSV',
                        (bundle.root / 'samples/sample_inputs.csv').read_bytes(),
                        file_name='example_traffic.csv', mime='text/csv')
                    st.download_button('Blank CSV template', input_template_csv(),
                        file_name='traffic_template.csv', mime='text/csv')
                    st.caption('Raw packet captures need the project preprocessing step first.')
            if batch is not None:
                _window_summary(batch)
                with st.expander('Look at the measurements'):
                    st.dataframe(batch.baseline.head(100), hide_index=True, width='stretch')
                    st.caption('The model also calculates four direction relationships from these values.')
            clicked = st.button('Predict traffic', type='primary', width='stretch',
                                disabled=batch is None, key='predict_traffic')
            _html('<div class="tw-how"><b>Traffic summary</b><span>→</span><b>Saved model</b><span>→</span><b>Prediction</b></div>')
    key = None if batch is None else prediction_key(batch.baseline, batch.sample_ids,
        bundle.manifest['models'][SELECTED_MODEL]['sha256'], batch.actual_labels)
    if clicked:
        try:
            result = predict(batch, bundle, SELECTED_MODEL)
            st.session_state['prediction_result'] = {'key': key, 'table': result}
        except (BundleError, DemoInputError, ValueError) as exc:
            st.error(str(exc))
    saved = st.session_state.get('prediction_result')
    result = saved['table'] if saved and saved['key'] == key else None
    with output_column:
        _html('<div class="tw-step">02 / Result</div><h2 class="tw-title">What does the model see?</h2>')
        if result is None:
            _html('<div class="tw-result empty">' + BOT +
                  '<h3>Ready when you are</h3><p>Choose an example and click <strong>Predict traffic</strong>.</p></div>')
        elif len(result) == 1:
            label = str(result.iloc[0].predicted_label)
            _prediction_card(label)
            if 'actual_label' in result:
                actual = str(result.iloc[0].actual_label)
                if actual == label:
                    st.caption(f'Matches the recorded label: {title(actual)}.')
                else:
                    st.warning(f'Recorded label: {title(actual)}. ' +
                        ('This is a false alarm.' if actual == 'BENIGN' else 'The model got this example wrong.'))
        else:
            flagged = int(result.predicted_label.ne('BENIGN').sum())
            c1, c2 = st.columns(2)
            c1.metric('Needs review', flagged)
            c2.metric('Predicted normal', len(result) - flagged)
            display = result[['sample_id', 'predicted_label']].rename(
                columns={'sample_id': 'Window', 'predicted_label': 'Prediction'})
            display['Prediction'] = display.Prediction.map(title)
            st.dataframe(display, hide_index=True, width='stretch', height=260)
            if 'actual_label' in result:
                with st.expander('Compare with the supplied labels'):
                    st.dataframe(result[['sample_id', 'actual_label', 'predicted_label', 'correct']],
                                 hide_index=True, width='stretch')
                    st.caption('This compares only these uploaded rows; it is not a new model benchmark.')
        if result is not None:
            st.download_button('Download prediction', result.to_csv(index=False).encode(),
                file_name='traffic_predictions.csv', mime='text/csv', width='stretch', key='result_download')
        st.caption('A flagged window is a prompt for review, not an automatic block.')


def _reset_replay() -> None:
    st.session_state['sim_cursor'] = 0
    st.session_state['sim_playing'] = False
    st.session_state['sim_last_tick'] = monotonic()
    st.session_state['sim_reviewed'] = []


def _simulation_source(bundle: LoadedBundle) -> pd.DataFrame:
    key = bundle.manifest['models'][SELECTED_MODEL]['sha256'] + '|office_replay_v1'
    cached = st.session_state.get('simulation_predictions')
    if cached is None or cached['key'] != key:
        # Sample selection chooses the story; labels never enter predict().
        ids = list(dict.fromkeys(scene[0] for scene in SCENES))
        source = predict(_subset_batch(bundle, ids), bundle, SELECTED_MODEL)
        table = replay_table(source)
        st.session_state['simulation_predictions'] = {'key': key, 'table': table}
        _reset_replay()
    return st.session_state['simulation_predictions']['table']


def _render_simulation(bundle: LoadedBundle) -> None:
    table = _simulation_source(bundle)
    if 'sim_cursor' not in st.session_state:
        _reset_replay()
    _html('<span class="tw-label">SIMULATION · SAVED TRAFFIC</span>'
          '<h2 class="tw-title" style="margin-top:.8rem">A little shift at the analyst’s desk</h2>'
          '<p class="tw-muted">Watch traffic summaries arrive, get a prediction and enter the review queue.</p>')
    _html('<div class="tw-network"><div class="tw-node"><div class="tw-node-icon">' + MONITOR +
          '</div><strong>Office devices</strong><small>Prepared traffic windows</small></div>'
          '<div class="tw-arrow">→</div><div class="tw-node">' + BOT +
          '<strong>Our classifier</strong><small>Predicts a traffic label</small></div>'
          '<div class="tw-arrow">→</div><div class="tw-node"><div class="tw-node-icon">' + MONITOR +
          '</div><strong>Analyst’s desk</strong><small>Reviews flagged windows</small></div></div>')
    st.caption('Real model predictions on selected saved examples. Device names, times and arrival order are fictional. No network is connected.')
    _replay_fragment(table)


def _replay_controls(table: pd.DataFrame) -> None:
    """One bounded replay. Timer ticks advance at most one row; pause stops it."""
    total = len(table)
    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    playing = st.session_state.get('sim_playing', False)
    if c1.button('Pause replay' if playing else 'Play simulation', type='primary',
                 width='stretch', key='sim_play'):
        if playing:
            st.session_state['sim_playing'] = False
        else:
            if st.session_state['sim_cursor'] >= total:
                _reset_replay()
            st.session_state['sim_playing'] = True
            st.session_state['sim_last_tick'] = monotonic()
            if st.session_state['sim_cursor'] == 0:
                st.session_state['sim_cursor'] = 1
        st.rerun()
    if c2.button('Next window', width='stretch', key='sim_next',
                 disabled=playing or st.session_state['sim_cursor'] >= total):
        st.session_state['sim_cursor'] += 1
        st.session_state['sim_last_tick'] = monotonic()
    if c3.button('Reset', width='stretch', key='sim_reset'):
        _reset_replay()
        st.rerun()
    if (st.session_state.get('sim_playing')
            and monotonic() - st.session_state.get('sim_last_tick', 0) >= 1.9):
        st.session_state['sim_cursor'] = min(total, st.session_state['sim_cursor'] + 1)
        st.session_state['sim_last_tick'] = monotonic()
        if st.session_state['sim_cursor'] == total:
            st.session_state['sim_playing'] = False
            st.rerun()
    cursor = st.session_state['sim_cursor']
    visible = table.iloc[:cursor]
    flagged = int(visible.needs_review.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric('Windows checked', f'{cursor} / {total}')
    c2.metric('Flagged for review', flagged)
    c3.metric('Predicted normal', cursor - flagged)
    st.progress(cursor / total)
    if not cursor:
        _html('<div class="tw-result empty"><h3>The office is waiting</h3>'
              '<p>Press Play to replay twelve windows, or use Next window to explain each step.</p></div>')
        return
    latest = visible.iloc[-1]
    story, queue = st.columns([1.1, 1.3], gap='large')
    with story:
        st.caption(f'LATEST WINDOW · {latest.time} · {latest.device} (simulated)')
        _prediction_card(latest.predicted_label)
        st.caption(analyst_note(latest.predicted_label))
        if latest.needs_review:
            reviewed = latest.event_id in st.session_state['sim_reviewed']
            if st.button('Reviewed ✓' if reviewed else 'Mark as reviewed', disabled=reviewed,
                         key=f'review_{latest.event_id}'):
                st.session_state['sim_reviewed'] = [*st.session_state['sim_reviewed'], latest.event_id]
                st.rerun()
    with queue:
        _html('<h3 class="tw-title">Activity feed</h3>')
        feed = visible.iloc[::-1].copy()
        feed['Status'] = ['Reviewed' if row.event_id in st.session_state['sim_reviewed']
                          else 'Needs review' if row.needs_review else 'No alert'
                          for row in feed.itertuples()]
        feed['Prediction'] = feed.predicted_label.map(title)
        st.dataframe(feed[['time', 'device', 'Prediction', 'Status']].rename(
            columns={'time': 'Demo time', 'device': 'Simulated device'}),
            hide_index=True, width='stretch', height=315)
    with st.expander('Did the model get these windows right?'):
        truth = visible[['time', 'device', 'actual_label', 'predicted_label']].copy()
        truth['Outcome'] = ['Correct' if row.actual_label == row.predicted_label else
                            'False alarm' if row.actual_label == 'BENIGN' else
                            'Missed attack' if row.predicted_label == 'BENIGN' else 'Wrong attack type'
                            for row in visible.itertuples()]
        truth['actual_label'] = truth.actual_label.map(title)
        truth['predicted_label'] = truth.predicted_label.map(title)
        st.dataframe(truth.rename(columns={'time': 'Demo time', 'device': 'Simulated device',
            'actual_label': 'Recorded label', 'predicted_label': 'Model prediction'}),
            hide_index=True, width='stretch')
        st.caption('Selected examples include mistakes. One normal example is deliberately replayed twice for the story; these twelve events are not an accuracy benchmark. Marking reviewed changes only this demo queue.')
    if cursor == total:
        st.caption('End of replay. The analyst still checks other evidence; a predicted normal window can contain a missed attack.')


if st is not None:
    _replay_fragment = st.fragment(run_every='2s')(_replay_controls)
else:
    _replay_fragment = _replay_controls


def _about_model(bundle: LoadedBundle) -> None:
    with st.expander('About this model & example files'):
        st.write('The selected C+T1 model is a Random Forest with 19 inputs: 15 extracted traffic measurements and four calculated direction relationships. It uses 80 trees, depth 20 and minimum leaf size 2. The app loads the saved model; it never fits or updates it.')
        metrics = bundle.manifest['models'][SELECTED_MODEL]['metrics']
        st.write(f"Recorded test: Macro F1 {metrics['test_macro_f1']:.2%} · accuracy {metrics['test_accuracy']:.2%} · benign false alarms {metrics['test_false_alarm_rate']:.2%}.")
        st.caption('The original test was previously examined for an earlier model, so this is not fresh independent confirmation. Heartbleed has no held-out score. Small rare-class support limits conclusions.')
        st.write('Examples come from 31 selected real validation rows. They include correct predictions and mistakes; they are not a new performance estimate. No original device names or addresses are published.')
        st.markdown('[Project reviewer](https://landotin.github.io/cicids2017-project-reviewer/) · [Code and evidence](https://github.com/Landotin/cicids2017-model-demo)')
        st.dataframe(pd.DataFrame([{'Measurement': name, 'Meaning': description}
            for name, description in FEATURE_DESCRIPTIONS.items()]), hide_index=True, width='stretch')
        for filename, label in [('sample_inputs.csv', 'Download example inputs'),
                                ('sample_labels.csv', 'Download example labels'),
                                ('sample_predictions.csv', 'Download recorded predictions')]:
            st.download_button(label, (bundle.root / 'samples' / filename).read_bytes(),
                               file_name=filename, mime='text/csv', key='about_' + filename)


def main() -> None:
    if st is None:
        raise RuntimeError('Install demo/requirements.txt before launching this Streamlit app.')
    st.set_page_config(page_title='Traffic Watch · Network attack demo', page_icon='🛡️', layout='wide')
    if st.__version__ != PINNED_STREAMLIT_VERSION:
        st.error(f'Use Streamlit {PINNED_STREAMLIT_VERSION} from demo/requirements.txt.')
        st.stop()
    st.html('<style>' + Path(__file__).with_name('style.css').read_text() + '</style>')
    _html('<header class="tw-header"><div><div class="tw-eyebrow">CICIDS2017 / Interactive model demo</div>'
          '<h1>Traffic Watch<span style="color:#52a294">.</span></h1>'
          '<p>A small window into how our model recognises network attacks.</p></div>'
          '<span class="tw-stamp">Saved model · Ready to predict</span></header>')
    try:
        bundle = load_cached_bundle(_bundle_path())
    except BundleError as exc:
        st.error(str(exc))
        st.stop()
    prediction_tab, simulation_tab = st.tabs(['Try a prediction', 'Workplace simulation'],
                                           key='demo_view', on_change='rerun')
    if prediction_tab.open:
        st.session_state['sim_playing'] = False
        with prediction_tab:
            _render_prediction(bundle)
    if simulation_tab.open:
        with simulation_tab:
            _render_simulation(bundle)
    _about_model(bundle)
    _html('<footer class="tw-footer"><span>A classroom demonstration using saved traffic windows.</span>'
          '<a href="https://landotin.github.io/cicids2017-project-reviewer/">Explore the project ↗</a></footer>')


if __name__ == '__main__':
    main()
