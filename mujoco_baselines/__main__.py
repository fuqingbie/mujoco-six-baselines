import sys


def _main():
    from . import doctor
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        return doctor.main()
    from .cli import main
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main())
