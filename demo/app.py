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
    @st.cache_resource(show_spinner="Checking the reviewed model bundle…")
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


def _metric_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.2%}" if 0 <= number <= 1 else f"{number:.4g}"


def _show_evidence(record: dict[str, Any], model_name: str) -> None:
    if model_name == "T1":
        st.info("T1 is the historical frozen model. Its earlier test result belongs to T1 only.")
    else:
        st.info("C+T1 is the selected 19-input model. Its test used the original partition previously examined for T1; it is not fresh independent confirmation.")
    metrics = record.get("metrics", {})
    fields = (
        ("test_macro_f1", "Test Macro F1"),
        ("test_accuracy", "Test accuracy"),
        ("test_false_alarm_rate", "Test benign false-alarm rate"),
        ("cv_macro_f1_mean", "Grouped CV Macro F1"),
        ("cv_macro_f1_std", "CV sample SD"),
        ("validation_macro_f1", "Validation Macro F1"),
        ("validation_accuracy", "Validation accuracy"),
        ("validation_false_alarm_rate", "Validation benign false-alarm rate"),
    )
    visible = [(label, _metric_text(metrics[key])) for key, label in fields if key in metrics]
    for start in range(0, len(visible), 3):
        group = visible[start:start + 3]
        columns = st.columns(len(group))
        for column, (label, value) in zip(columns, group):
            column.metric(label, value)


def _sample_batch(bundle: LoadedBundle) -> InputBatch:
    return validate_input_table(
        bundle.sample_inputs.assign(actual_label=bundle.sample_labels.to_numpy()),
        class_labels=bundle.class_labels,
    )


def _uploaded_batch(uploaded: Any, class_labels: tuple[str, ...]) -> InputBatch:
    return parse_uploaded_csv(uploaded.getvalue(), class_labels=class_labels)


def main() -> None:
    if st is None:
        raise RuntimeError("Install demo/requirements.txt before launching this Streamlit app.")
    st.set_page_config(page_title="CICIDS2017 Flow Demo", page_icon="🛡️", layout="centered")
    if st.__version__ != PINNED_STREAMLIT_VERSION:
        st.error(
            f"This demo expects Streamlit {PINNED_STREAMLIT_VERSION}; "
            f"the current runtime is {st.__version__}. Install demo/requirements.txt."
        )
        st.stop()
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {background: #f3f7f8; border-radius: 12px; padding: .7rem 1rem;}
        .small-note {color: #53636b; font-size: .95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Traffic window classifier")
    st.caption("A saved-model demonstration for prepared 60-second conversation windows.")
    st.warning("This is not live network monitoring. Packet-to-window preparation happens before this app.")

    try:
        bundle = load_cached_bundle(_bundle_path())
    except BundleError as exc:
        st.error(str(exc))
        st.stop()

    models_record = bundle.manifest["models"]
    st.info("C+T1 is the selected study model: mean grouped CV Macro F1 "
            "was 72.30%, versus T1's 71.43%. The gain is small across three folds. "
            "C+T1 reached 87.31% test Macro F1 without refitting. This test was already examined for T1, so it is not fresh independent evidence.")
    with st.expander("Download the reviewed example files"):
        st.caption("These 31 selected validation rows include correct predictions and mistakes. "
                   "They are examples for the presentation, not a performance benchmark.")
        for filename, title in (("sample_inputs.csv", "Download example measurements"),
                                ("sample_labels.csv", "Download actual labels"),
                                ("sample_predictions.csv", "Download reviewed predictions")):
            st.download_button(title, data=(bundle.root / "samples" / filename).read_bytes(),
                               file_name=filename, mime="text/csv", width="stretch")
    source = st.selectbox("Choose input", ("Reviewed example", "Upload a prepared CSV"))
    batch: InputBatch | None = None
    if source == "Reviewed example":
        batch = _sample_batch(bundle)
        sample_note = bundle.root / "samples" / "SAMPLE_DATA.md"
        st.caption("Small reviewed validation examples. They illustrate model behavior and are not a benchmark.")
        if sample_note.is_file():
            with st.expander("Example provenance"):
                st.markdown(sample_note.read_text(encoding="utf-8"))
    else:
        st.markdown(
            "Upload rows prepared from **60-second conversation windows**. Start with the 15-column "
            "template; add one row per window. Optional `actual_label` or `label` is used only for comparison."
        )
        st.download_button(
            "Download 15-measurement CSV template",
            data=input_template_csv(),
            file_name="prepared_window_inputs_template.csv",
            mime="text/csv",
            width="stretch",
        )
        uploaded = st.file_uploader(
            "Upload measurements from prepared windows",
            type=("csv",),
            help=(f"CSV size limit: {MAX_UPLOAD_BYTES:,} bytes; "
                  f"row limit: {MAX_UPLOAD_ROWS:,}. Include the fifteen baseline columns. "
                  "Known IDs, labels, and metadata are ignored; other extra columns are rejected."),
        )
        if uploaded is not None:
            try:
                batch = _uploaded_batch(uploaded, bundle.class_labels)
            except DemoInputError as exc:
                st.error(str(exc))

    feature_descriptions = pd.DataFrame(
        [{"measurement": name, "meaning": FEATURE_DESCRIPTIONS[name]} for name in BASE_FEATURES]
    )
    st.subheader("The fifteen baseline measurements")
    with st.expander("View names and short descriptions"):
        st.dataframe(feature_descriptions, hide_index=True, width="stretch")

    st.subheader("Choose a saved model")
    selected = st.selectbox(
        "Model",
        MODEL_NAMES,
        index=1,
        format_func=lambda name: "C+T1 · 19 inputs · selected" if name == "CT1"
        else "T1 · 15 inputs · historical",
    )
    _show_evidence(models_record[selected], selected)
    st.caption(f"Evidence status: `{models_record[selected]['evidence_status']}`")
    st.caption("Prediction uses only the defined measurements. Labels, sample IDs, and other metadata are excluded.")

    if batch is None:
        st.stop()

    st.subheader(f"Input preview · {len(batch.baseline):,} rows")
    st.dataframe(batch.baseline.head(100), hide_index=True, width="stretch")
    if len(batch.baseline) > 100:
        st.caption("Showing the first 100 rows. Predictions cover every validated row.")

    reveal = False
    if selected == "CT1":
        reveal = st.checkbox("Show the four calculated C inputs")
    if reveal:
        try:
            transformed = prepare_model_inputs(batch, selected)
            st.dataframe(transformed.loc[:, list(ENGINEERED_FEATURES)].head(100),
                         hide_index=True, width="stretch")
        except (BundleError, DemoInputError) as exc:
            st.error(str(exc))
            st.stop()

    if st.button("Predict", type="primary", width="stretch"):
        try:
            predictions = predict(batch, bundle, selected)
        except (BundleError, DemoInputError, ValueError) as exc:
            st.error(str(exc))
            st.stop()
        st.subheader("Predictions")
        st.dataframe(predictions, hide_index=True, width="stretch")
        if "correct" in predictions:
            correct = int(predictions["correct"].sum())
            st.caption(f"{correct:,} of {len(predictions):,} supplied labels matched. "
                       "This checks these rows only; it is not a new performance estimate.")
        st.download_button(
            "Download prediction table",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected.lower()}_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    st.caption("Clicking Predict runs inference only. The app never fits or updates a model.")


if __name__ == "__main__":
    main()
