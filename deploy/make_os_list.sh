#!/usr/bin/env bash
# Chromapi: write deploy/os_list.json, the Raspberry Pi Imager repository that offers the SD image with Imager's own
# customisation step (hostname, user, Wi-Fi, SSH), as for Ubuntu Server. "Use custom" on the .img.xz has no such step.
#
#   deploy/make_os_list.sh chromapi.img.xz [url of the image]
#
# In Imager: App options > Content repository > Use custom URL (the raw URL of os_list.json) or Use custom file.
# From a terminal: rpi-imager --repo /absolute/path/of/os_list.json
set -euo pipefail

IMAGE="${1:?usage: $0 chromapi.img.xz [url]}"
DEPLOY="$(cd "$(dirname "$0")" && pwd)"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$DEPLOY/../pyproject.toml")"
URL="${2:-https://github.com/Mowibox/Chromapi/releases/download/v$VERSION/chromapi.img.xz}"
OUT="$DEPLOY/os_list.json"

echo "Calculating checksums of $IMAGE..."
python3 - "$OUT" <<PY
import json, sys

def device(name, tag, description, matching):  # tags and matching types of Imager's official list
    return {"name": name, "tags": [f"{tag}-64bit", f"{tag}-32bit"], "description": description,
            "icon": f"https://downloads.raspberrypi.com/imager/icons/RPi_{name.split('Pi ')[1].replace(' ', '_')}.png",
            "matching_type": matching, "capabilities": []}

json.dump({
    "imager": {"devices": [
        device("Raspberry Pi 4", "pi4", "Raspberry Pi 4 Model B", "inclusive") | {"default": True},
        device("Raspberry Pi 5", "pi5", "Raspberry Pi 5", "exclusive"),
        device("Raspberry Pi Zero 2 W", "pi3", "Raspberry Pi Zero 2 W", "inclusive"),
    ]},
    "os_list": [{
        "name": "Chromapi OS $VERSION",
        "description": "Ubuntu Server 24.04 LTS with Chromapi SDK",
        "icon": "https://raw.githubusercontent.com/Mowibox/Chromapi/main/src/chromapi/web/static/logo.png",
        "url": "$URL",
        "extract_size": $(xz --robot -l "$IMAGE" | awk '/^totals/ {print $5}'),
        "extract_sha256": "$(xz -dc -T0 "$IMAGE" | sha256sum | cut -d' ' -f1)",
        "image_download_size": $(stat -c %s "$IMAGE"),
        "image_download_sha256": "$(sha256sum "$IMAGE" | cut -d' ' -f1)",
        "release_date": "$(date -r "$IMAGE" +%F)",
        "init_format": "cloudinit",  # Imager writes user-data and network-config, read by cloud-init at first boot
        "devices": ["pi4-64bit", "pi5-64bit", "pi3-64bit"],
        "capabilities": [],
    }],
}, open(sys.argv[1], "w"), indent=2)
PY
echo "Wrote $OUT ($URL)"
