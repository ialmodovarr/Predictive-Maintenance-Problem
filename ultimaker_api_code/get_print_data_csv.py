import hashlib
import re
import secrets
import requests
import json
import time
import csv
from datetime import datetime
from pathlib import Path

PRINTER = "http://172.27.135.1"

USERNAME = "6bf926c99944ad9e15da49c2ed1bffb5"
PASSWORD = "716b548893f2fe16f0d1929b94857484d291b8af67e61685ba9c1ad24f682e30"

def digest_get(uri):
    """GET printer data. Try normal GET first; use SHA-256 Digest if required."""

    # 1. Try without authentication
    response = requests.get(PRINTER + uri)

    # If it works, return the data
    if response.ok:
        return response.json() # returns json data into python dictionary

    # 2. If authentication is required, get the Digest challenge
    if response.status_code != 401:
        response.raise_for_status()

    challenge = response.headers["WWW-Authenticate"]

    realm = re.search(r'realm="([^"]+)"', challenge).group(1)
    nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
    qop = re.search(r'qop="([^"]+)"', challenge).group(1)

    # 3. Create Digest values
    nc = "00000001"
    cnonce = secrets.token_hex(16)

    HA1 = hashlib.sha256(
        f"{USERNAME}:{realm}:{PASSWORD}".encode()
    ).hexdigest()

    HA2 = hashlib.sha256(
        f"GET:{uri}".encode()
    ).hexdigest()

    response_hash = hashlib.sha256(
        f"{HA1}:{nonce}:{nc}:{cnonce}:{qop}:{HA2}".encode()
    ).hexdigest()

    authorization = (
        f'Digest username="{USERNAME}", '
        f'realm="{realm}", '
        f'nonce="{nonce}", '
        f'uri="{uri}", '
        f'algorithm=SHA-256, '
        f'response="{response_hash}", '
        f'qop={qop}, '
        f'nc={nc}, '
        f'cnonce="{cnonce}"'
    )

    # 4. Make authenticated request
    response = requests.get(
        PRINTER + uri,
        headers={"Authorization": authorization}
    )

    response.raise_for_status()
    return response.json()

# --------------------------------------------------
# FLATTEN NESTED JSON INTO CSV COLUMNS
# --------------------------------------------------
def flatten_json(data, prefix=""):
    flattened = {}

    if isinstance(data, dict):
        if not data:
            flattened[prefix] = "{}"

        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten_json(value, new_key))

    elif isinstance(data, list):
        if not data:
            flattened[prefix] = "[]"

        for index, value in enumerate(data):
            new_key = f"{prefix}.{index}"
            flattened.update(flatten_json(value, new_key))

    else:
        flattened[prefix] = data

    return flattened


# --------------------------------------------------
# SAVE CSV ROW, ADDING COLUMNS IF NECESSARY
# --------------------------------------------------
def save_csv_row(output_file, row):
    existing_columns = []
    existing_rows = []

    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []

            # Only load existing rows if new columns are discovered
            new_columns = [key for key in row if key not in existing_columns]

            if new_columns:
                existing_rows = list(reader)
    else:
        new_columns = list(row.keys())

    if new_columns:
        columns = existing_columns + new_columns

        # Rewrite the header to accommodate newly discovered fields
        temp_file = output_file.with_suffix(".tmp")

        with open(temp_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(existing_rows)

        temp_file.replace(output_file)

    else:
        columns = existing_columns

    # Append the latest API readings
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)
        file.flush()


# --------------------------------------------------
# GET ALL PRINTER DATA
# --------------------------------------------------
def main():
    output_file = Path(__file__).with_name("printer_data_all.csv")

    while True:
        try:

            start_time = time.perf_counter()

            # Request print job data
            job = digest_get("/api/v1/print_job")
            job_timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            # Request printer data
            printer = digest_get("/api/v1/printer")
            printer_timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            # Start each row with acquisition timestamps
            row = {
                "job_timestamp": job_timestamp,
                "printer_timestamp": printer_timestamp
            }

            # Automatically extract EVERY field
            row.update(flatten_json(job, "job"))
            row.update(flatten_json(printer, "printer"))

            # Write all fields to CSV
            save_csv_row(output_file, row)

        except Exception as e:
            print("Error collecting data:", e)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        sleep_time = max(0, 1 - elapsed_time)
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()