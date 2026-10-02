"""Run the whole freight-rate pipeline with one command.

    python run_pipeline.py              # load -> CV selection -> holdout -> unseen-city check
                                        # -> final fit -> both prediction files -> score.py -> chart
    python run_pipeline.py --skip-eval  # final fit, prediction files, score.py and chart only

With ``--skip-eval`` the setup comes from ``reports/metrics.json`` (written by
the last full run) or, if that file is missing, from ``src.model.FINAL_SPEC``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip CV, holdout and unseen-city evaluation; only fit, predict, write and score")
    parser.add_argument("--no-score", action="store_true", help="do not run score.py at the end")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # keep log lines in order when output is redirected

    from src import config as cfg
    from src import data, predict
    from src import train as T
    from src.model import FINAL_SPEC, ModelSpec

    t0 = time.perf_counter()
    print("== Load and check data")
    ds = data.load_all()
    print(f"train {len(ds.train):,} rows, validation {len(ds.validation):,} rows, {len(ds.city_table)} cities")

    if args.skip_eval:
        saved = T.load_metrics().get("chosen", {}).get("spec")
        spec = ModelSpec.from_dict(saved) if saved else FINAL_SPEC
        print(f"== Evaluation skipped; setup from {'reports/metrics.json' if saved else 'model.FINAL_SPEC'}")
    else:
        print("== Evaluation (CV selection, holdout, unseen-city check)")
        spec = T.run_evaluation(ds)
    print(f"setup: {spec.label()}")

    print("== Final fit on all clean Jan-Oct rows, predictions, files")
    predict.run_final(spec, ds)

    if not args.no_score:
        print("== score.py")
        cmd = [sys.executable, "score.py", "--predictions", "validation_predictions.csv",
               "--december-predictions", "data/december_chart_inputs.csv"]
        result = subprocess.run(cmd, cwd=cfg.ROOT)
        if result.returncode != 0:
            print("score.py FAILED")
            return result.returncode
    print(f"== Done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
