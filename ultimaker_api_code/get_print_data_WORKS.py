import hashlib
import re
import secrets
import requests
import json

PRINTER = "http://172.27.135.1"

USERNAME = "6bf926c99944ad9e15da49c2ed1bffb5"
PASSWORD = "716b548893f2fe16f0d1929b94857484d291b8af67e61685ba9c1ad24f682e30"

def digest_get(uri):
    """GET printer data. Try normal GET first; use SHA-256 Digest if required."""

    # 1. Try without authentication
    response = requests.get(PRINTER + uri)

    # If it works, return the data
    if response.ok:
        return response.json()

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
# GET PRINTER DATA
# --------------------------------------------------

endpoints = [
    "/api/v1/print_job",
    "/api/v1/printer",
]

for endpoint in endpoints:
    print("\n" + "=" * 60)
    print(endpoint)
    print("=" * 60)

    try:
        data = digest_get(endpoint)
        print(json.dumps(data, indent=2))

    except requests.HTTPError as e:
        print("HTTP error:", e)
        print("Response:", e.response.text)

    except Exception as e:
        print("Error:", e)