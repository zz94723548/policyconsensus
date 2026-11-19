"""Audits for the frozen final-test protocol; no model or simulator execution."""
import json
from pathlib import Path

from modular_policy.common.error_moments import file_hash


def read_final_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol['purpose'] != 'frozen_final_offline_error_moments' or protocol['risk_mode'] != 'shared':
        raise ValueError('Invalid final-test protocol')
    if not 0 <= protocol['fixed_rgb_weight'] <= 1:
        raise ValueError('Invalid frozen fixed weight')
    for name in ['rgb_checkpoint', 'pcd_checkpoint', 'statistics_checkpoint', 'calibration',
                 'source_manifest', 'risk_split']:
        record = protocol['files'][name]
        if file_hash(record['path']) != record['sha256']:
            raise ValueError(f'Frozen final-test file changed: {name}')
    return protocol
