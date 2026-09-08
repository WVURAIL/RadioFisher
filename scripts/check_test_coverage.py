#!/usr/bin/env python3
"""Require branch measurement and a floor for every supported Python module."""
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    report = Path(sys.argv[1] if len(sys.argv) > 1 else 'coverage.json')
    data = json.loads(report.read_text())
    errors = []
    if not data.get('meta', {}).get('branch_coverage'):
        errors.append('branch coverage must be enabled')
    total = data['totals']['percent_covered']
    if total < 85:
        errors.append(f'overall coverage {total:.2f}% is below 85%')
    measured = {Path(key).as_posix(): value for key, value in data['files'].items()}
    for path in sorted((root/'radiofisher').rglob('*.py')):
        relative = path.relative_to(root).as_posix()
        record = measured.get(relative, measured.get(path.as_posix()))
        if record is None:
            errors.append(f'{relative}: missing from coverage report')
            continue
        percent = record['summary']['percent_covered']
        if percent < 80:
            errors.append(f'{relative}: {percent:.2f}% is below 80%')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Coverage passed: {total:.2f}% overall, every module >=80%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
