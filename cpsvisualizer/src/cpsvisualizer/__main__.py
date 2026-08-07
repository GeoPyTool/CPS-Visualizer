import sys
from cpsvisualizer.app import main as gui
from cpsvisualizer.app_cli import main as cli
from cpsvisualizer.web import main as web

if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli()
    elif "--web" in sys.argv:
        web()
    else:
        gui()