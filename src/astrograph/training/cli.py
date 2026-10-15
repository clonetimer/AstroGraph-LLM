"""Auditable CLI for small-model data preparation, preflight and fixed-set evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .small_model import TASKS, TrainingExample, prepare_training, preflight, evaluate_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--tool-root', type=Path, required=True)
    p.add_argument('--python', type=Path, help='Explicit isolated training interpreter')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare-generated', help='Validate user-authorized generated examples without assigning human approval')
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--python', type=Path, help='Explicit isolated training interpreter')
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--tool-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    p = sub.add_parser('evaluate')
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('review-export', help='Validate completed human review against a frozen draft pack')
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare-generated':
            from .self_instruct import GeneratedExample, prepare_generated
            examples = [GeneratedExample.model_validate_json(line) for line in args.examples.read_text(encoding='utf-8').splitlines() if line.strip()]
            result = prepare_generated(examples, output=args.output, source_root=args.source_root)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.command == 'review-export':
            from .review_pack import export_reviewed
            result = export_reviewed(args.pack, args.reviews, args.output)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.command == 'preflight':
            result = preflight(args.tool_root.resolve(), args.model.resolve(), args.python)
        else:
            examples = [TrainingExample.model_validate_json(line) for line in args.examples.read_text(encoding='utf-8').splitlines() if line.strip()]
            if args.command == 'prepare':
                result = prepare_training(examples, output=args.output, task=args.task,
                                          tool_root=args.tool_root, model=args.model, gpu=args.gpu, python_executable=args.python)
                print(json.dumps(result, ensure_ascii=False))
                return 0
            result = evaluate_predictions(examples, json.loads(args.predictions.read_text(encoding='utf-8')))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
