import sys
from cpsvisualizer.app import main as gui
from cpsvisualizer.app_cli import main as cli

if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli()
    else:
        gui()