#!/usr/bin/env bash
# Chromapi: clean an installed card before it is copied into an SD image, then power off. On the robot, as the user
# that ran setup_rpi.sh, from the SSH login shell:
#
#   unset HISTFILE; ./chromapi/deploy/clean_image.sh
#
# unset HISTFILE: the login shell would write its history back when the session closes. The robot is unreachable
# afterwards (no Wi-Fi,
# no SSH key, password locked): take the card out and copy it on a PC before it boots again (see the end).
# Made again at the image's first boot: SSH host keys, hostname and chromapi-name.txt (chromapi-boot), machine-id
# (systemd), user, password, SSH key and Wi-Fi from Raspberry Pi Imager (cloud-init), access code, session key and
# HTTPS certificate (chromapi).
set -euo pipefail

[ "$(id -u)" -ne 0 ] || { echo "Run as the robot's user, not root (sudo is called when needed)." >&2; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BOOT=/boot/firmware

echo "Image user: $USER (uid $(id -u))."
[ "$(id -u)" -eq 1000 ] || echo "[!] uid is not 1000: chromapi-boot will not give it the Imager password." >&2
if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    echo "[!] $REPO has local changes, they go into the image:" >&2
    git -C "$REPO" status --short >&2
fi
echo "Repository: $(git -C "$REPO" describe --tags --always --dirty)"
echo
echo "This removes the Wi-Fi, SSH keys, password, calibration, gamepad pairing and logs of this robot, then powers off."
read -rp "Type 'clean' to go on: " answer
[ "$answer" = clean ] || { echo "Nothing done."; exit 1; }

sudo systemctl stop "chromapi@$USER" "chromapi-fan@$USER" 2>/dev/null || true  # they write ~/.config/chromapi

# This robot's own data: access code, session key, TLS key, calibration, recordings, reports, history, caches
rm -rf "$HOME/.config/chromapi" "$HOME/chromapi_recordings" "$HOME"/chromapi_report_*.txt "$HOME/.ssh" \
    "$HOME/.cache" "$HOME/.gitconfig" "$HOME/.git-credentials" "$HOME"/.*_history "$HOME/.lesshst" "$HOME/.viminfo"
sudo rm -rf /root/.ssh /root/.*_history
sudo passwd -l "$USER"  # Imager's password is given back at the first boot (chromapi-boot)

# Identity: SSH host keys, gamepad link keys, hostname (chromapi-boot names it chromapi-xxxx)
sudo rm -f /etc/ssh/ssh_host_*
sudo rm -rf /var/lib/bluetooth/*
echo chromapi | sudo tee /etc/hostname >/dev/null
sudo sed -i 's/^127\.0\.1\.1.*/127.0.1.1 chromapi/' /etc/hosts
sudo rm -f "$BOOT/chromapi-name.txt"

# Networks (Wi-Fi passwords) and cloud-init: it runs again at the first boot and reads Imager's files
sudo cloud-init clean --logs --seed --machine-id --configs all
sudo rm -f /etc/netplan/*.yaml /var/lib/chromapi/network-config.sha256
sudo rm -rf /etc/NetworkManager/system-connections/*
# Written back by chromapi-boot after the first boot: until then cloud-init may set the name of user-data
sudo rm -f /etc/cloud/cloud.cfg.d/99-chromapi.cfg
sudo tee "$BOOT/user-data" >/dev/null <<EOF
#cloud-config
# Chromapi image: read once, at the first boot. The Wi-Fi is in network-config, next to this file.
# Nothing to change here to use the web app. To log in over SSH (user "$USER", no password by default), choose a
# password and remove the "#" of these lines. Spaces only, no tabs.
#ssh_pwauth: true
#chpasswd:
#  expire: false
#  users:
#    - name: $USER
#      password: choose-a-password
#      type: text
# Name of the robot on the network (default: chromapi-xxxx, written in chromapi-name.txt after the first boot):
#hostname: my-chromapi
EOF
sudo tee "$BOOT/network-config" >/dev/null <<'EOF'
# Netplan configuration, applied at the first boot by cloud-init, and after it by Chromapi each time this file is
# edited (put the card in a PC to change the Wi-Fi). Spaces only, no tabs. https://netplan.io/reference
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: true
      optional: true
#  wifis:
#    wlan0:
#      dhcp4: true
#      optional: true
#      access-points:
#        "my home wifi":
#          password: "S3kr1t"
EOF

# Size of the image
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
sudo journalctl --rotate && sudo journalctl --vacuum-time=1s
sudo find /var/log -type f \( -name '*.gz' -o -name '*.[0-9]' \) -delete
sudo find /var/log -type f -exec truncate -s 0 {} +

cat <<'EOF'

Clean. Powering off: take the card out and do not boot it in a robot before it is copied. On the PC:
  lsblk                                                  # the card, e.g. /dev/sdX
  sudo dd if=/dev/sdX of=chromapi.img bs=4M status=progress conv=fsync
  sudo pishrink.sh -aZ chromapi.img                      # gives chromapi.img.xz
EOF
sudo systemctl poweroff
