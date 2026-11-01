#!/usr/bin/env python3
"""Entry point for ``python -m toffee.amputate.canary``.

Subcommands are forwarded to the module CLIs::

    python -m toffee.amputate.canary prep  --tokenizer ... --domain-pack ...
    python -m toffee.amputate.canary run   --model dense=... --model amp=...
    python -m toffee.amputate.canary selftest
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "prep":
        from toffee.amputate.canary.prep_corpus import main as prep_main
        return prep_main(rest)
    if cmd == "run":
        from toffee.amputate.canary.run_canary import main as run_main
        return run_main(rest)
    if cmd == "selftest":
        from toffee.amputate.canary.prep_corpus import run_selftest as p
        from toffee.amputate.canary.run_canary import run_selftest as r
        ok = p() == 0 and r() == 0
        return 0 if ok else 1
    print(f"unknown canary command {cmd!r} (expected prep|run|selftest)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
