"""Small smoke-test for packaged exe behavior.

This script prints the resolved Config.DATA_DIR and attempts two writes:
 - a relative path 'data/pack_test_rel.txt' (to simulate old behavior)
 - a path under Config.DATA_DIR (the desired, user-writable location)

Run as: python tools/pack_test.py
When packaged, running the exe should still succeed writing to Config.DATA_DIR.
"""

import os
import sys

# Ensure project root is on sys.path when run from dist or sources
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.config import Config
except Exception as e:
    print("ERROR: could not import Config:", e)
    # Fallback: try to read MEWS_DATA_DIR or use tempdir
    cfg_base = os.environ.get('MEWS_DATA_DIR') or os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'mews-fin')
    class Dummy:
        DATA_DIR = cfg_base
    Config = Dummy


def try_write(path, content):
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"OK: wrote {path}")
    except Exception as e:
        print(f"FAIL: {path} -> {repr(e)}")


def main():
    print('Config.DATA_DIR=', Config.DATA_DIR)

    rel = os.path.join('data', 'pack_test_rel.txt')
    print('\nAttempting relative write to', rel)
    try_write(rel, 'relative')

    cfgp = os.path.join(Config.DATA_DIR, 'pack_test_cfg.txt')
    print('\nAttempting Config.DATA_DIR write to', cfgp)
    try_write(cfgp, 'configdir')

    print('\nDone')


if __name__ == '__main__':
    main()
