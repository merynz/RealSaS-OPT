"""Deprecated compatibility shim.

The implementation is family-generic. Keep this entrypoint only so historical
commands do not silently break; all current workflows must call
run_family_iris_fit_v1.py directly.
"""

from experiments.first_family_fit_v1.run_family_iris_fit_v1 import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
