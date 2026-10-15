"""Separate-process upload worker; shares the canonical job store and runtime."""
from __future__ import annotations
import argparse
from pathlib import Path
from .upload_report import UploadReportJobStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args(argv)
    result = UploadReportJobStore(args.root).run(args.job_id)
    return 0 if result['status'] == 'SUCCEEDED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
