"""Plain-language labels and the explicitly fictional office replay layout."""
from __future__ import annotations
import hashlib
import json
import pandas as pd

SELECTED_MODEL = 'CT1'
EXAMPLES = {
    'validation-000000': 'A normal traffic window',
    'validation-458072': 'A high-traffic attack',
    'validation-406529': 'A scanning attempt',
    'validation-068045': 'A tricky traffic window',
}
# Aliases and time slots are fictional. IDs point to the unchanged validation examples.
SCENES = (
    ('validation-000000', 'Office laptop', 'Workstations'),
    ('validation-085341', 'File server', 'Server room'),
    ('validation-406529', 'Web server', 'Server room'),
    ('validation-458072', 'Web server', 'Server room'),
    ('validation-068045', 'Office laptop', 'Workstations'),
    ('validation-131213', 'Admin workstation', 'Workstations'),
    ('validation-373214', 'Office desktop', 'Workstations'),
    ('validation-299537', 'Web server', 'Server room'),
    ('validation-331867', 'File server', 'Server room'),
    ('validation-000000', 'Office laptop', 'Workstations'),
    ('validation-181987', 'Web server', 'Server room'),
    ('validation-459353', 'Web server', 'Server room'),
)
TITLES = {
    'BENIGN': 'Normal traffic', 'Bot': 'Bot activity', 'DDoS': 'Distributed denial of service',
    'PortScan': 'Port scanning', 'FTP-Patator': 'FTP password attack',
    'SSH-Patator': 'SSH password attack', 'DoS GoldenEye': 'Denial of service · GoldenEye',
    'DoS Hulk': 'Denial of service · Hulk', 'DoS Slowhttptest': 'Denial of service · Slow HTTP',
    'DoS slowloris': 'Denial of service · Slowloris', 'Infiltration': 'Infiltration',
    'Web Attack – Brute Force': 'Web password attack',
    'Web Attack – Sql Injection': 'SQL injection', 'Web Attack – XSS': 'Cross-site scripting',
}


def title(label: str) -> str:
    return TITLES.get(str(label), str(label))


def prediction_key(baseline, sample_ids, model_sha: str, actual_labels=None) -> str:
    """An old prediction must not appear beside a different input or model."""
    digest = hashlib.sha256(model_sha.encode())
    digest.update(pd.util.hash_pandas_object(baseline, index=False).to_numpy().tobytes())
    digest.update(json.dumps(list(map(str, sample_ids)), ensure_ascii=False).encode())
    # Labels never enter inference, but a changed supplied label must refresh its comparison.
    if actual_labels is not None:
        digest.update(json.dumps(list(map(str, actual_labels)), ensure_ascii=False).encode())
    return digest.hexdigest()


def replay_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Decorate real predictions with scenario-only event IDs, times and device aliases."""
    if predictions.sample_id.duplicated().any():
        raise ValueError('Replay source predictions need unique sample IDs.')
    source = predictions.set_index('sample_id')
    records = []
    for index, (sample_id, device, area) in enumerate(SCENES):
        if sample_id not in source.index:
            raise ValueError(f'Missing reviewed replay example: {sample_id}')
        row = source.loc[sample_id]
        records.append(dict(event_id=f'window-{index + 1:02d}', time=f'09:{index:02d}',
            sample_id=sample_id, device=device, area=area,
            predicted_label=str(row.predicted_label), actual_label=str(row.actual_label),
            needs_review=str(row.predicted_label) != 'BENIGN'))
    return pd.DataFrame(records)


def analyst_note(label: str) -> str:
    """Suggested human follow-up, not an explanation of the forest's decision."""
    if label == 'BENIGN':
        return 'No alert from this model. Continue routine checks; a normal prediction can still be wrong.'
    if label == 'PortScan':
        return 'Check connection logs and whether the source is an approved scanner.'
    if label.startswith('DoS') or label == 'DDoS':
        return 'Check request rates, service health and connection logs before deciding what to do.'
    if label in ('FTP-Patator', 'SSH-Patator') or 'Brute Force' in label:
        return 'Check sign-in failures and whether the activity is expected.'
    return 'Compare this alert with host, application and connection logs before taking action.'
