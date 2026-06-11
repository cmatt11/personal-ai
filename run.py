#!/usr/bin/env python3
"""Entry point for the personal AI.

Usage:
  python run.py                 start the terminal chat
  python run.py --web [port]    start the local web UI (default port 8000)
  python run.py --doctor        check environment readiness
  python run.py --setup         run the first-time setup wizard
"""

import sys

from personal_ai.config import Config


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in ("--web", "-w"):
        from personal_ai.web_ui import serve

        port = 8000
        for a in args[1:]:
            if a.isdigit():
                port = int(a)
        serve(port=port)
        return 0
    if args and args[0] == "--doctor":
        from personal_ai.setup_wizard import doctor

        print(doctor(Config()))
        return 0
    if args and args[0] == "--setup":
        from personal_ai.setup_wizard import run_wizard

        run_wizard(Config())
        return 0

    from personal_ai.cli import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
