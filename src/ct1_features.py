"""Self-contained input contract for the bounded C+T1 follow-up model."""

import numpy as np
import pandas as pd


BASE_FEATURES = (
    'protocol', 'duration_seconds', 'packet_count', 'total_bytes',
    'mean_packet_bytes', 'min_packet_bytes', 'max_packet_bytes',
    'forward_packet_count', 'backward_packet_count', 'forward_bytes',
    'backward_bytes', 'syn_count', 'ack_count', 'fin_count', 'rst_count',
)

ENGINEERED_FEATURES = (
    'forward_backward_packet_ratio',
    'forward_backward_byte_ratio',
    'forward_packet_share',
    'no_backward_packets',
)


def make_input(table, engineered=True):
    """Return ordered 15- or 19-column inputs, with the recorded C formulas.

    The two forward/backward ratios are zero when their denominator is zero,
    matching ``reference_models.make_input``.  Packet share needs a positive
    packet count because its denominator is required to be meaningful.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError('Model input must be a pandas DataFrame.')
    if not table.columns.is_unique:
        raise ValueError('Model input columns must be unique.')
    missing = [name for name in BASE_FEATURES if name not in table.columns]
    if missing:
        raise ValueError(f'Missing baseline model inputs: {missing}.')

    features = table.loc[:, BASE_FEATURES].copy()
    try:
        values = features.to_numpy(dtype='float64')
    except (TypeError, ValueError) as exc:
        raise ValueError('Baseline model inputs must be numeric.') from exc
    if not np.isfinite(values).all():
        raise ValueError('Model input contains a non-finite value.')

    if engineered:
        packet_count = features['packet_count'].to_numpy(dtype='float64')
        if (packet_count <= 0).any():
            raise ValueError('packet_count must be positive to calculate packet share.')

        for name, numerator, denominator in (
            ('forward_backward_packet_ratio', 'forward_packet_count', 'backward_packet_count'),
            ('forward_backward_byte_ratio', 'forward_bytes', 'backward_bytes'),
        ):
            a = features[numerator].to_numpy(dtype='float64')
            b = features[denominator].to_numpy(dtype='float64')
            features[name] = np.divide(a, b, out=np.zeros_like(a), where=b > 0)
        features['forward_packet_share'] = (
            features['forward_packet_count'].to_numpy(dtype='float64') / packet_count
        )
        features['no_backward_packets'] = features['backward_packet_count'].eq(0).astype('int8')

    expected = list(BASE_FEATURES) + (list(ENGINEERED_FEATURES) if engineered else [])
    if features.columns.tolist() != expected:
        raise ValueError('The input feature order differs from the CT1 contract.')
    if not np.isfinite(features.to_numpy(dtype='float64')).all():
        raise ValueError('Engineered model input contains a non-finite value.')
    return features
