#!/usr/bin/env -S uv run python
"""zenn.sh を使って Scrap の全ページをひとつの JSON として出力する。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def get_scrap(scrap: str) -> dict:
    zenn = Path(__file__).resolve().with_name("zenn.sh")
    page = 1
    visited = set()
    result = None
    comments = []

    while page is not None:
        if type(page) is not int or page < 1 or page in visited:
            raise ValueError("不正な next_page を受信しました")
        visited.add(page)
        response = subprocess.run(
            [str(zenn), "scrap", "get", scrap, "--page", str(page),
             "--count", "20", "--machine-readable"],
            check=True,
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        data = json.loads(response.stdout)
        if not isinstance(data, dict) or not isinstance(data.get("comments"), list):
            raise ValueError("レスポンスに comments 配列がありません")
        if "scrap" not in data or "next_page" not in data:
            raise ValueError("レスポンスに scrap または next_page がありません")
        if result is None:
            result = data
        comments.extend(data["comments"])
        page = data["next_page"]

    result["comments"] = comments
    result["next_page"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scrap", help="Scrap の slug または URL")
    args = parser.parse_args()
    try:
        result = get_scrap(args.scrap)
    except subprocess.CalledProcessError as error:
        print("Scrap の取得に失敗しました", file=sys.stderr)
        return error.returncode
    except (OSError, ValueError) as error:
        print(f"Scrap の取得に失敗しました: {error}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
