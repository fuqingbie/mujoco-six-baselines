"""CLI entry: python -m mujoco_baselines {doctor|run|report|smoke} ..."""
import sys


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "doctor":
        from . import doctor
        return doctor.main()
    if cmd == "run":
        from .run import runner
        return runner.main(argv[1:])
    if cmd == "report":
        from . import report
        return report.main(argv[1:])
    if cmd == "smoke":
        from . import smoke
        return smoke.main(argv[1:])
    print(f"unknown command {cmd}")
    return 1
