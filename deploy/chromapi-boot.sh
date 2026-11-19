#!/usr/bin/env bash
# Chromapi, at each boot, before the network and SSH (chromapi-boot.service). Installed as /usr/local/sbin/chromapi-boot.
#  - Hostname: a default name (cloned card) becomes chromapi-<last 4 of the Pi's serial> so two robots have two names.
#  - Wi-Fi: /boot/firmware/network-config edited on a PC since the last boot is applied (cloud-init only reads it on
#    the first boot), in a netplan file of its own: the networks added by hand in the others are kept, netplan merges.
#  - The name is in /boot/firmware/chromapi-name.txt: put the card in any PC to read it (no network, no screen).
#  - SSH: host keys missing (removed from an SD image) are made again, unique to this robot.
set -u
DEFAULT_NAMES=" ubuntu raspberrypi chromapi "
NETWORK_CONFIG=/boot/firmware/network-config
STAMP=/var/lib/chromapi/network-config.sha256
NETPLAN=/etc/netplan/60-chromapi-network-config.yaml  # never 50-cloud-init.yaml: it may hold networks added by hand
NAME_FILE=/boot/firmware/chromapi-name.txt

host="$(cat /etc/hostname 2>/dev/null)"
if [[ "$DEFAULT_NAMES" == *" $host "* ]]; then
    serial="$(tr -d '\0' </sys/firmware/devicetree/base/serial-number 2>/dev/null || awk '/^Serial/ {print $3}' /proc/cpuinfo)"
    if [ -n "$serial" ]; then
        new="chromapi-${serial: -4}"
        echo "$new" >/etc/hostname
        hostname "$new"
        if grep -q '^127\.0\.1\.1' /etc/hosts; then
            sed -i "s/^127\.0\.1\.1.*/127.0.1.1 $new/" /etc/hosts
        else
            echo "127.0.1.1 $new" >>/etc/hosts
        fi
        echo "Hostname: $host -> $new"
    fi
fi

host="$(cat /etc/hostname)"
if [ -d "$(dirname "$NAME_FILE")" ] && [ "$(cat "$NAME_FILE" 2>/dev/null)" != "$host" ]; then  # written on change only
    echo "$host" >"$NAME_FILE"
fi

# Wi-Fi passphrases -> their WPA-PSK key (64 hex digits, accepted by netplan as is): the card read on a PC no longer
# shows a passphrase reused elsewhere. The key still joins that one network. In place, comments kept; before the
# change check below, so an edited file is applied already hashed. WPA3 (SAE) needs the passphrase: left as is.
for file in "$NETWORK_CONFIG" /etc/netplan/50-cloud-init.yaml; do
    [ -f "$file" ] || continue
    python3 - "$file" <<'PY' || echo "Network: could not hash the Wi-Fi passphrases of $file" >&2
import hashlib, os, re, sys, yaml
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
config = yaml.safe_load(text)
config = config.get("network", config) if isinstance(config, dict) else {}
new = text
for iface in (config.get("wifis") or {}).values():
    for ssid, ap in ((iface or {}).get("access-points") or {}).items():
        ap = ap or {}
        auth = ap.get("auth") if isinstance(ap.get("auth"), dict) else {}
        password = auth.get("password", ap.get("password"))
        if not isinstance(password, str) or not 8 <= len(password) <= 63:
            continue  # open network, or already a 64-hex key
        if auth.get("key-management", "psk") != "psk":
            continue
        psk = hashlib.pbkdf2_hmac("sha1", password.encode(), str(ssid).encode(), 4096, 32).hex()
        new = re.sub(r"(password:[ \t]*)([\"']?)" + re.escape(password) + r"\2[ \t]*$",
                     lambda m: f'{m.group(1)}"{psk}"', new, flags=re.M)
if new != text:
    with open(path + ".new", "w", encoding="utf-8") as f:
        f.write(new)
    try:
        os.chmod(path + ".new", os.stat(path).st_mode & 0o777)  # 600 for netplan; FAT has no modes
    except OSError:
        pass
    os.replace(path + ".new", path)
    print(f"Network: Wi-Fi passphrases of {path} replaced by their key")
PY
done

if [ -f "$NETWORK_CONFIG" ]; then
    mkdir -p "$(dirname "$STAMP")"
    sum="$(sha256sum "$NETWORK_CONFIG" | cut -d' ' -f1)"
    if [ ! -f "$STAMP" ]; then
        echo "$sum" >"$STAMP"  # first run: the network in use already comes from this file
    elif [ "$sum" != "$(cat "$STAMP")" ]; then
        # network-config is netplan v2, with or without its "network:" key (cloud-init adds it)
        if python3 - "$NETWORK_CONFIG" >"$NETPLAN.new" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
if not isinstance(config, dict) or not config:
    sys.exit(1)  # only comments: keep the network as it is
yaml.safe_dump(config if "network" in config else {"network": config}, sys.stdout, sort_keys=False)
PY
        then
            # netplan's generator already ran at boot on the old file: generate again, and systemd reloads the units
            chmod 600 "$NETPLAN.new" && mv "$NETPLAN.new" "$NETPLAN" && netplan generate && systemctl daemon-reload \
                && echo "Network: $NETWORK_CONFIG applied"
        else
            rm -f "$NETPLAN.new"
            echo "Network: $NETWORK_CONFIG unreadable or empty, network unchanged" >&2
        fi
        echo "$sum" >"$STAMP"
    fi
fi

ssh-keygen -A  # only the missing keys

if [ ! -e /var/lib/cloud/instance/boot-finished ]; then
    # First boot of an SD image (clean_image.sh). An Imager that customises it (1.x does, 2.x only for its own list)
    # writes a password hash in user-data: cloud-init skips it for a user that already exists, so it is set here
    user="$(getent passwd 1000 | cut -d: -f1)"
    [ -z "$user" ] || python3 - /boot/firmware/user-data "$user" <<'PY' | chpasswd -e
import sys, yaml
data = yaml.safe_load(open(sys.argv[1])) or {}
for entry in [data.get("user")] + (data.get("users") or []):  # Imager writes "user:" (cloud-init's default user)
    if isinstance(entry, dict) and entry.get("name") == sys.argv[2] and entry.get("passwd"):
        print(f"{sys.argv[2]}:{entry['passwd']}")
PY
elif [ ! -f /etc/cloud/cloud.cfg.d/99-chromapi.cfg ]; then
    # After cloud-init's first run (it set the name of user-data, if any): it would put that name back at each boot
    echo "preserve_hostname: true" >/etc/cloud/cloud.cfg.d/99-chromapi.cfg
fi
