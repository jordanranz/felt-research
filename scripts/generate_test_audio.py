"""Generate inspectable fixtures using the installed felt research package."""

import argparse
import json

from felt.audio.fixtures import export_fixtures
from felt.data import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment001.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = export_fixtures(load_config(args.config), args.out, args.split, args.count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
