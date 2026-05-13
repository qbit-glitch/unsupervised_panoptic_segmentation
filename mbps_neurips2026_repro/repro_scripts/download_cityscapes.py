#!/usr/bin/env python3
"""Download Cityscapes dataset files using requests with proper session auth.

Usage:
    export CITYSCAPES_USERNAME="your@email.com"
    export CITYSCAPES_PASSWORD="yourpassword"
    python scripts/download_cityscapes.py --output_dir /path/to/save
"""

import argparse
import os
import sys

import requests

LOGIN_URL = "https://www.cityscapes-dataset.com/login/"
DOWNLOAD_BASE = "https://www.cityscapes-dataset.com/file-handling/?packageID="

# Package IDs for Cityscapes dataset files
PACKAGES = {
    "gtFine_trainvaltest.zip": 1,
    "leftImg8bit_trainvaltest.zip": 3,
}


def login(username: str, password: str) -> requests.Session:
    """Login to Cityscapes and return authenticated session."""
    session = requests.Session()

    # Get CSRF token from login page
    resp = session.get(LOGIN_URL)
    resp.raise_for_status()

    # Extract csrfmiddlewaretoken from the page
    csrf_token = None
    for line in resp.text.split("\n"):
        if "csrfmiddlewaretoken" in line and 'value="' in line:
            csrf_token = line.split('value="')[1].split('"')[0]
            break

    if not csrf_token:
        # Try from cookies
        csrf_token = session.cookies.get("csrftoken", "")

    # Login
    login_data = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token,
        "submit": "Login",
    }
    headers = {
        "Referer": LOGIN_URL,
    }

    resp = session.post(LOGIN_URL, data=login_data, headers=headers)

    # Check if login succeeded (should redirect or show no error)
    if "login" in resp.url.lower() and resp.status_code == 200:
        # Check if we got redirected back to login (failure)
        if "incorrect" in resp.text.lower() or "invalid" in resp.text.lower():
            print("ERROR: Login failed - check username/password")
            sys.exit(1)

    print(f"Logged in as {username}")
    return session


def download_file(session: requests.Session, package_name: str, package_id: int,
                  output_dir: str):
    """Download a single Cityscapes package."""
    output_path = os.path.join(output_dir, package_name)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        if size_mb > 10:  # Real zip should be > 10MB
            print(f"  {package_name} already exists ({size_mb:.0f}MB), skipping")
            return True
        else:
            print(f"  {package_name} exists but too small ({size_mb:.1f}MB), re-downloading")
            os.remove(output_path)

    url = f"{DOWNLOAD_BASE}{package_id}"
    print(f"  Downloading {package_name} (packageID={package_id})...")

    resp = session.get(url, stream=True)
    resp.raise_for_status()

    # Check content type - should be application/zip, not text/html
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        print(f"  ERROR: Got HTML instead of zip. Authentication may have failed.")
        print(f"  Content-Type: {content_type}")
        print(f"  First 200 chars: {resp.text[:200]}")
        return False

    total_size = int(resp.headers.get("Content-Length", 0))
    total_mb = total_size / (1024 * 1024) if total_size else 0

    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 128):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = downloaded / total_size * 100
                    dl_mb = downloaded / (1024 * 1024)
                    print(f"\r  {dl_mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)

    print(f"\n  Saved: {output_path} ({downloaded / (1024*1024):.0f}MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Cityscapes dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--packages", nargs="*", default=list(PACKAGES.keys()),
                        help="Package names to download")
    args = parser.parse_args()

    username = os.environ.get("CITYSCAPES_USERNAME")
    password = os.environ.get("CITYSCAPES_PASSWORD")

    if not username or not password:
        print("ERROR: Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD env vars")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Login
    session = login(username, password)

    # Download each package
    for pkg_name in args.packages:
        if pkg_name not in PACKAGES:
            print(f"Unknown package: {pkg_name}")
            continue
        success = download_file(session, pkg_name, PACKAGES[pkg_name], args.output_dir)
        if not success:
            print(f"Failed to download {pkg_name}")
            sys.exit(1)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
