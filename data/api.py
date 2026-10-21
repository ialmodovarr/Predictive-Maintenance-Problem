import hashlib
import re
import secrets
import requests
import json
import time
from datetime import datetime
from pathlib import Path

'''

Collects the API data live every 1 second (can be changed)
Writes to a date and time stamped api_data_xxx.csv every 5 seconds (can be changed)\

'''


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


# request to send commands to printer
def digest_put(uri, payload):
    """Send an authenticated PUT request to the printer API."""

    # First, try the request without authentication.
    # A protected endpoint should respond with 401 and a Digest challenge.
    response = requests.put(
        PRINTER + uri,
        json=payload,
        timeout=5
    )

    if response.ok:
        return response

    if response.status_code != 401:
        response.raise_for_status()

    # Read the printer's authentication challenge.
    challenge = response.headers["WWW-Authenticate"]

    realm = re.search(r'realm="([^"]+)"', challenge).group(1)
    nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
    qop = re.search(r'qop="([^"]+)"', challenge).group(1)

    # Create the Digest authentication values.
    nc = "00000001"
    cnonce = secrets.token_hex(16)

    HA1 = hashlib.sha256(
        f"{USERNAME}:{realm}:{PASSWORD}".encode()
    ).hexdigest()

    # IMPORTANT: Use PUT here, not GET.
    HA2 = hashlib.sha256(
        f"PUT:{uri}".encode()
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

    # Retry the PUT request with authentication.
    response = requests.put(
        PRINTER + uri,
        json=payload,
        headers={"Authorization": authorization},
        timeout=5
    )

    response.raise_for_status()
    return response

# stops printer for extreme extruder values 
def stop_printer():
    """Attempt to abort printing and turn off extruder 1 heating."""

    # Command 1: Abort the current print job.
    abort_uri = "/api/v1/print_job/state"

    # Command 2: Set extruder 1's hotend target to 0 degrees C.
    temperature_uri = (
        "/api/v1/printer/heads/0/extruders/0/"
        "hotend/temperature/target"
    )

    try:
        response = digest_put(
            abort_uri,
            {"target": "abort"}
        )
        print(f"Print-abort request returned HTTP {response.status_code}.")

        response = digest_put(
            temperature_uri,
            0.0
        )
        print(
            f"Hotend target request returned HTTP {response.status_code}."
        )

    except requests.RequestException as exc:
        print(f"CRITICAL: Printer stop command failed: {exc}")
        print("Check the printer directly and stop it manually if needed.")


# --------------------------------------------------
# GET PRINTER DATA
# --------------------------------------------------
# Create a new output file every 10 seconds
FILE_ROTATION_SECONDS = 10

POLL_INTERVAL_SECONDS = 1

def main():
    endpoints = [
        "/api/v1/print_job",
        "/api/v1/printer",
    ]

    # Get the folder containing your Python script (data)
    data_dir = Path(__file__).resolve().parent

    # Navigate into api_data
    api_dir = data_dir / "api_data"

    # Create the folder if it doesn't exist
    api_dir.mkdir(parents=True, exist_ok=True)


    output_file = None
    file = None
    next_rotation = 0

    try:
        while True:
            current_time = time.monotonic()

            # Create a new file when needed
            if file is None or current_time >= next_rotation:

                # Close the previous file before creating a new one
                if file is not None:
                    file.close()

                # Generate a timestamped filename
                run_timestamp = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S-%f"
                )

                output_file = (
                    api_dir / f"api_data_{run_timestamp}.txt"
                )

                file = open(
                    output_file,
                    "w",
                    encoding="utf-8",
                    buffering=1
                )

                file.write(
                    f"Printer data collection started: {datetime.now()}\n"
                )
                file.write("=" * 60 + "\n")

                print(f"Writing to: {output_file}")

                # Schedule the next file rotation
                next_rotation = (
                    time.monotonic() + FILE_ROTATION_SECONDS
                )

            # Collect printer data
            for endpoint in endpoints:
                header = (
                    "\n" + "=" * 60 + "\n"
                    + endpoint + "\n"
                    + "=" * 60
                )

                print(header, file=file)

                try:
                    data = digest_get(endpoint)

                    timestamp = datetime.now().strftime(
                        "%H:%M:%S.%f"
                    )[:-3]

                    output = (
                        f"Laptop timestamp: {timestamp}\n"
                        + json.dumps(data, indent=2)
                    )

                except requests.HTTPError as e:
                    output = (
                        f"HTTP error: {e}\n"
                        f"Response: {e.response.text}"
                    )

                except Exception as e:
                    output = f"Error: {e}"

                print(output, file=file, flush=True)

            # Wait before the next polling cycle
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nData collection stopped.")

    finally:
        if file is not None:
            file.close()
            print(f"Closed file: {output_file}")


if __name__ == "__main__":
    main()


# Your data here

# ============================================================
# /api/v1/print_job
# ============================================================
# Laptop timestamp: 11:24:34.798
# {
#   "datetime_cleaned": "",
#   "datetime_finished": "",
#   "datetime_started": "2026-10-14T14:51:47+00:00",
#   "material_0_usage": 2763.0,
#   "material_1_usage": -1.0,
#   "name": "UMS3_flow_sensor_v2_plate.ufp",
#   "pause_source": "",
#   "progress": 0.18,
#   "reprint_original_uuid": "",
#   "result": "",
#   "source": "WEB_API",
#   "source_application": "Ultimaker Connect",
#   "source_user": "",
#   "state": "printing",
#   "time_elapsed": 1688,
#   "time_estimated": 9274,
#   "time_total": 9262,
#   "used_hotend_types_0": [
#     {
#       "serial": "4535bf390000",
#       "type": "AA 0.4"
#     }
#   ],
#   "used_hotend_types_1": [
#     {
#       "serial": "0cbd393a0000",
#       "type": "AA 0.4"
#     }
#   ],
#   "used_materials_0_guid": [
#     {
#       "UUID": "506c9f0d-e3aa-4bd4-b2d2-23e2425b1aa9",
#       "name": "Generic PLA Generic"
#     }
#   ],
#   "used_materials_1_guid": [],
#   "uuid": "01dd9860-6c0d-420d-849e-0aa2cb52e4b7"
# }

# ============================================================
# /api/v1/printer
# ============================================================
# Laptop timestamp: 11:24:35.387
# {
#   "bed": {
#     "pre_heat": {
#       "active": false


#     }, BED TEMP
#     "temperature": {
#       "current": 60.0,
#       "target": 60.0
#     },


#     "type": "glass"
#   },
#   "diagnostics": {},
#   "heads": [
#     {
# ==================== PRINT HEAD ====================


#       PRINT HEAD ACCELERATION
#       "acceleration": 1500.0,


#       "extruders": [
#         {

# ==================== EXTRUDER 1 ====================
#           "active_material": {
#             "GUID": "506c9f0d-e3aa-4bd4-b2d2-23e2425b1aa9",
#             "guid": "506c9f0d-e3aa-4bd4-b2d2-23e2425b1aa9",
#             "length_remaining": -1.0
#           },

#           FEEDER DATA 
#           "feeder": {
#             "acceleration": 3000.0,
#             "jerk": 5.0,
#             "max_speed": 45.0
#           },


#           "hotend": {
#             "id": "AA 0.4",
#             "offset": {
#               "state": "valid",
#               "x": 0.0,
#               "y": 0.0,
#               "z": 0.0
#             },
#             "revision": "1",
#             "serial": "4535bf390000",
#             "statistics": {
#               "last_material_guid": "506c9f0d-e3aa-4bd4-b2d2-23e2425b1aa9",

#               EXTRUDER DATA
#               "material_extruded": 845450,
#               "max_temperature_exposed": 253,
#               "prints_since_cleaned": "100",
#               "time_spent_hot": 3956220


#             },

#               HOTEND TEMPERATURE
#             "temperature": {
#               "current": 200.0,
#               "target": 200.0


#             }
#           }
#         },
#         {
        
    # ==================== EXTRUDER 2 ====================

#           "active_material": {
#             "GUID": "7e6207c4-22ff-441a-b261-ff89f166d6a0",
#             "guid": "7e6207c4-22ff-441a-b261-ff89f166d6a0",
#             "length_remaining": -1.0
#           },
#           "feeder": {
#             "acceleration": 3000.0,
#             "jerk": 5.0,
#             "max_speed": 45.0
#           },
#           "hotend": {
#             "id": "AA 0.4",
#             "offset": {
#               "state": "valid",
#               "x": -0.7317073170731714,
#               "y": 0.0,
#               "z": 0.0
#             },
#             "revision": "1",
#             "serial": "0cbd393a0000",
#             "statistics": {
#               "last_material_guid": "7e6207c4-22ff-441a-b261-ff89f166d6a0",
#               "material_extruded": 497560,
#               "max_temperature_exposed": 273,
#               "prints_since_cleaned": "61",
#               "time_spent_hot": 3371040
#             },
#             "temperature": {
#               "current": 39.7,
#               "target": 0.0
#             }
#           }
#         }
#       ],


    # ==================== PRINT HEAD ====================
#       "fan": 100.0,
#       "jerk": {
#         "x": 30.0,
#         "y": 30.0,
#         "z": 0.4
#       },
#       "max_speed": {
#         "x": 300.0,
#         "y": 300.0,
#         "z": 40.0
#       },
#       "position": {
#         "x": 130.089,
#         "y": 102.147,
#         "z": 2.0
#       }
#     }


#   ],
#   "led": {
#     "blink": {},
#     "brightness": 100.0,
#     "hue": 0.0,
#     "saturation": 0.0
#   },
#   "network": {
#     "ethernet": {
#       "connected": false,
#       "enabled": false
#     },
#     "wifi": {
#       "connected": true,
#       "enabled": true,
#       "mode": "WIRELESS",
#       "ssid": "UM-NO-HOTSPOT-NAME-SET"
#     },
#     "wifi_networks": [
#       {
#         "SSID": "wahoo",
#         "connected": true,
#         "security_required": false,
#         "strength": 72
#       },
#       {
#         "SSID": "UVA Guest",
#         "connected": false,
#         "security_required": false,
#         "strength": 71
#       },
#       {
#         "SSID": "UVA WiFi Setup",
#         "connected": false,
#         "security_required": false,
#         "strength": 71
#       },
#       {
#         "SSID": "eduroam",
#         "connected": false,
#         "security_required": true,
#         "strength": 71
#       },
#       {
#         "SSID": "DIRECT-vcHL-L8360CDW_BRf7b4",
#         "connected": false,
#         "security_required": true,
#         "strength": 52
#       }
#     ]
#   },
#   "status": "printing",
#   "validate_header": {}
# }

# ============================================================
# /api/v1/print_job
# ============================================================
# Laptop timestamp: 11:24:36.471
# {
#   "datetime_cleaned": "",
#   "datetime_finished": "",
#   "datetime_started": "2026-10-14T14:51:47+00:00",
#   "material_0_usage": 2763.0,
#   "material_1_usage": -1.0,
#   "name": "UMS3_flow_sensor_v2_plate.ufp",
#   "pause_source": "",
#   "progress": 0.18,
#   "reprint_original_uuid": "",
#   "result": "",
#   "source": "WEB_API",
#   "source_application": "Ultimaker Connect",
#   "source_user": "",
#   "state": "printing",
#   "time_elapsed": 1688,
#   "time_estimated": 9274,
#   "time_total": 9262,
#   "used_hotend_types_0": [
#     {
#       "serial": "4535bf390000",
#       "type": "AA 0.4"
#     }
#   ],
#   "used_hotend_types_1": [
#     {
#       "serial": "0cbd393a0000",
#       "type": "AA 0.4"
#     }
#   ],
#   "used_materials_0_guid": [
#     {
#       "UUID": "506c9f0d-e3aa-4bd4-b2d2-23e2425b1aa9",
#       "name": "Generic PLA Generic"
#     }
#   ],
#   "used_materials_1_guid": [],
#   "uuid": "01dd9860-6c0d-420d-849e-0aa2cb52e4b7"
# }
