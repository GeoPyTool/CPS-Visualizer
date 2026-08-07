"""``cpsv`` command-line dispatcher.

Default (no sub-command) runs the batch CLI.  ``cpsv gui`` launches the
PySide6 desktop GUI.  ``cpsv web`` starts the Flask web interface.

Examples::

    cpsv --help
    cpsv "Ag.csv Cu.csv" "log_transform equalize_hist Euclidean" silent
    cpsv gui
    cpsv web --port 6789
    cpsv web --host 0.0.0.0 --port 8000
"""
import sys


USAGE = """\
cpsv - CPS-Visualizer command line

Usage:
  cpsv [DATA_FILES] [FUNCTIONS] [MODE]      Run the batch CLI (default).
  cpsv gui                                  Launch the desktop GUI.
  cpsv web [--host HOST] [--port PORT]      Launch the web interface.

CLI arguments:
  DATA_FILES    space-separated list of CSV/XLSX data files (one quoted arg)
  FUNCTIONS    space-separated transforms and distance metrics (one quoted arg)
  MODE         show (default) | silent   (silent saves PNG/PDF/SVG)

Web options:
  --host HOST   bind address (default 127.0.0.1)
  --port PORT   bind port    (default 5005)

Examples:
  cpsv "Ag.csv Cu.csv" "log_transform equalize_hist Euclidean" silent
  cpsv gui
  cpsv web --port 6789
"""


def _parse_web_args(argv):
    host = '127.0.0.1'
    port = 5005
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--host',):
            if i + 1 < len(argv):
                host = argv[i + 1]
                i += 2
                continue
        elif a.startswith('--host='):
            host = a.split('=', 1)[1]
            i += 1
            continue
        elif a in ('--port',):
            if i + 1 < len(argv):
                try:
                    port = int(argv[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
        elif a.startswith('--port='):
            try:
                port = int(a.split('=', 1)[1])
            except ValueError:
                pass
            i += 1
            continue
        i += 1
    return host, port


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(USAGE)
        return 0

    cmd = argv[0]
    if cmd == 'gui':
        from cpsvisualizer.app import main as gui_main
        return gui_main()
    if cmd == 'web':
        host, port = _parse_web_args(argv[1:])
        from cpsvisualizer.web import main as web_main
        return web_main(host=host, port=port)
    if cmd in ('cli',):
        argv = argv[1:]

    # Default: batch CLI.
    from cpsvisualizer.app_cli import main as cli_main
    return cli_main(*_cli_args(argv))


def _cli_args(argv):
    """Map raw argv into (data_files, functions, mode) for app_cli.main."""
    data_files = None
    functions = None
    mode = 'show'
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--mode',) and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
            continue
        if a.startswith('--mode='):
            mode = a.split('=', 1)[1]
            i += 1
            continue
        positional.append(a)
        i += 1
    if len(positional) >= 1:
        data_files = positional[0]
    if len(positional) >= 2:
        functions = positional[1]
    if len(positional) >= 3:
        mode = positional[2]
    return data_files, functions, mode


if __name__ == '__main__':
    sys.exit(main())