#!/usr/bin/env python3
"""Download NVD CVE JSON 2.0 yearly feeds, skipping feeds that haven't changed.

Usage:
    python src/fetch_nvd.py                    # 2015 to current year
    python src/fetch_nvd.py --start 2020       # custom range
    python src/fetch_nvd.py --force            # re-download everything
"""
import argparse
import datetime as dt
import gzip
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://nvd.nist.gov/feeds/json/cve/2.0"
HEADERS = {"User-Agent": "CVEscore-fetcher/1.0"}


def get(url: str, retries: int = 4) -> bytes:
    """GET with exponential backoff (NVD sometimes returns 403/503 under load)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * 2**attempt
            print(f"    retry in {wait}s ({e})")
            time.sleep(wait)


def parse_meta(text: str) -> dict:
    # .meta files look like "sha256:ABC123...", one key per line
    return {k.strip(): v.strip() for k, v in
            (line.split(":", 1) for line in text.strip().splitlines() if ":" in line)}


def fetch_feed(name: str, out: Path, force: bool) -> bool:
    """Download one feed. Returns True if a new file was written."""
    gz_path = out / f"nvdcve-2.0-{name}.json.gz"
    meta_path = out / f"nvdcve-2.0-{name}.meta"

    remote_meta = get(f"{BASE}/nvdcve-2.0-{name}.meta").decode()
    remote_sha = parse_meta(remote_meta)["sha256"].lower()

    if not force and gz_path.exists() and meta_path.exists():
        local_sha = parse_meta(meta_path.read_text()).get("sha256", "").lower()
        if local_sha == remote_sha:
            print(f"  {name}: up to date")
            return False

    print(f"  {name}: downloading...")
    data = get(f"{BASE}/nvdcve-2.0-{name}.json.gz")

    # The meta checksum refers to the uncompressed JSON; accept either to be safe
    if remote_sha not in (hashlib.sha256(gzip.decompress(data)).hexdigest(),
                          hashlib.sha256(data).hexdigest()):
        raise ValueError("checksum mismatch (feed may be mid-update, rerun later)")

    # Write to a temp file then rename, so an interrupted run never leaves a corrupt feed
    tmp = out / f"{gz_path.name}.part"
    tmp.write_bytes(data)
    tmp.replace(gz_path)
    meta_path.write_text(remote_meta)
    print(f"  {name}: {len(data) / 1e6:.1f} MB saved")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=dt.date.today().year)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--sleep", type=float, default=3.0, help="pause between feeds (be polite to NVD)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    updated, failed = [], []

    print(f"Fetching NVD feeds {args.start}-{args.end} into {args.out}/")
    for year in range(args.start, args.end + 1):
        try:
            if fetch_feed(str(year), args.out, args.force):
                updated.append(year)
        except Exception as e:
            print(f"  {year}: FAILED ({e})")
            failed.append(year)
        time.sleep(args.sleep)

    print(f"\nDone. Updated: {updated or 'none'}. Failed: {failed or 'none'}.")
    return 1 if failed else 0   # non-zero exit code so cron/CI notices failures


if __name__ == "__main__":
    sys.exit(main())
