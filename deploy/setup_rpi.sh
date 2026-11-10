#!/usr/bin/env bash
# Chromapi set-up of a Raspberry Pi 4, 5 or Zero 2W on Ubuntu Server 24.04, without a prompt. Safe to run again.
#
#   git clone https://github.com/Mowibox/chromapi.git && ./chromapi/deploy/setup_rpi.sh && sudo reboot
#
# Run as the robot's user (not root): its venv, groups and services. Then, after the reboot:
#   chromapi flash      # the firmware shipped with this version
#   chromapi health     # check what is still wrong
# and the web app at https://<hostname>.local:5000.
set -euo pipefail

[ "$(id -u)" -ne 0 ] || { echo "Run as the robot's user, not root (sudo is called when needed)." >&2; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$REPO/deploy"
VENV="$HOME/.venvs/chromapi"
BOOT=/boot/firmware
step() { printf '\n== %s\n' "$*"; }

step "Packages"
sudo apt-get update
# stm32flash, gpiod: firmware flash. libportaudio2, alsa-utils: audio. bluez: controller. avahi: <hostname>.local and
# chromapi.local. openssl: HTTPS certificate. iw: Wi-Fi power saving.
sudo apt-get install -y git python3-venv stm32flash gpiod libportaudio2 alsa-utils bluez avahi-daemon avahi-utils openssl iw
# vcgencmd (support report only): optional
sudo apt-get install -y libraspberrypi-bin || echo "libraspberrypi-bin not available: the support report has no vcgencmd"

step "Boot configuration ($BOOT/config.txt)"
CONFIG="$BOOT/config.txt"
MODEL="$(tr -d '\0' </proc/device-tree/model)"
echo "Board: $MODEL"
[ -e "$CONFIG.chromapi.bak" ] || sudo cp "$CONFIG" "$CONFIG.chromapi.bak"
# Legacy camera stack (/dev/video0) needs the fake KMS driver; the Pi 5 has libcamera only
# (camera_auto_detect: Ubuntu's default loads the sensor's libcamera driver, which takes it from the legacy stack)
case "$MODEL" in "Raspberry Pi 5"*) ;; *) sudo sed -i -e 's/^dtoverlay=vc4-kms-v3d/dtoverlay=vc4-fkms-v3d/' \
    -e 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CONFIG" ;; esac
sudo sed -i '/^gpu_mem=/d' "$CONFIG"
# Lines an older set-up appended under [all]: now in the board sections below
for line in enable_uart=1 start_x=1 gpu_mem=128 dtoverlay=pwm,pin=13,func=4 dtoverlay=miniuart-bt core_freq_min=500 \
    dtoverlay=googlevoicehat-soundcard; do
    sudo sed -i "/^${line//\//\\/}\$/d" "$CONFIG"
done
# Chromapi's block, rewritten on each run, per board ([pi4], [pi5], [pi02]):
#  - miniuart-bt: ttyAMA0 to the STM32; core_freq_min keeps the mini-UART's baud rate (Bluetooth) steady
#  - pwm: the fan on GPIO 13 (not checked on a Pi 5)
#  - googlevoicehat-soundcard: speaker and microphone (I2S)
sudo sed -i '/^# chromapi begin$/,/^# chromapi end$/d' "$CONFIG"
sudo tee -a "$CONFIG" >/dev/null <<'BLOCK'
# chromapi begin
[all]
enable_uart=1
dtoverlay=googlevoicehat-soundcard
[pi4]
start_x=1
gpu_mem=128
dtoverlay=pwm,pin=13,func=4
dtoverlay=miniuart-bt
core_freq_min=500
[pi02]
start_x=1
gpu_mem=128
dtoverlay=pwm,pin=13,func=4
dtoverlay=miniuart-bt
core_freq_min=400
[pi5]
dtparam=uart0=on
[all]
# chromapi end
BLOCK
# No Linux console on the bridge's serial port
sudo sed -i -E 's/console=(serial0|ttyAMA0),[0-9]+ ?//g' "$BOOT/cmdline.txt"

step "Groups"
sudo groupadd -f gpio
sudo groupadd -f chromapi
# dialout: serial port. gpio: PWM, flash. plugdev, input: controller. video: camera. audio. chromapi: power off.
sudo usermod -aG dialout,gpio,plugdev,input,video,audio,chromapi "$USER"

step "udev, polkit and ALSA"
sudo rm -f /etc/udev/rules.d/99-pwm.rules  # older name of 99-chromapi-pwm.rules
sudo install -m 644 "$DEPLOY/99-chromapi-pwm.rules" "$DEPLOY/99-chromapi-gamepad.rules" /etc/udev/rules.d/
# Wi-Fi power saving delays packets (~100 ms): the page's sticks stutter
echo 'ACTION=="add", SUBSYSTEM=="net", KERNEL=="wlan*", RUN+="/usr/sbin/iw dev %k set power_save off"' \
    | sudo tee /etc/udev/rules.d/70-chromapi-wifi-powersave.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo install -m 644 "$DEPLOY/50-chromapi-power.rules" /etc/polkit-1/rules.d/
sudo install -m 644 "$DEPLOY/asound.conf" /etc/asound.conf

step "Python ($VENV)"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$REPO[audio,rl]"  # editable: a git pull updates the code
mkdir -p "$HOME/.local/bin" && ln -sf "$VENV/bin/chromapi" "$HOME/.local/bin/chromapi"  # on PATH at next login

step "Services"
sudo install -m 644 "$DEPLOY/chromapi-audio-keepalive.service" /etc/systemd/system/
sudo install -m 755 "$DEPLOY/chromapi-audio-check.sh" /usr/local/sbin/chromapi-audio-check  # its ExecStartPre
for unit in chromapi@.service chromapi-fan@.service; do  # the venv's Python instead of the system one
    sed "s|/usr/bin/python3|$VENV/bin/python|g" "$DEPLOY/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
# At each boot: default hostname -> chromapi-xxxx, network-config edited on a PC applied, missing SSH host keys made
sudo install -m 755 "$DEPLOY/chromapi-boot.sh" /usr/local/sbin/chromapi-boot
sudo install -m 644 "$DEPLOY/chromapi-boot.service" /etc/systemd/system/
# chromapi.local alias
sudo install -m 755 "$DEPLOY/chromapi-alias.sh" /usr/local/sbin/chromapi-alias
sudo install -m 644 "$DEPLOY/chromapi-alias.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable chromapi-boot chromapi-alias chromapi-audio-keepalive "chromapi-fan@$USER" "chromapi@$USER" avahi-daemon
# No unattended apt upgrade
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
# Stay on Ubuntu 24.04 LTS
[ ! -f /etc/update-manager/release-upgrades ] || sudo sed -i 's/^Prompt=.*/Prompt=never/' /etc/update-manager/release-upgrades

sudo /usr/local/sbin/chromapi-boot  # now: the name given below is the robot's

step "Done"
echo "Reboot (sudo reboot): boot configuration, groups and services apply then."
echo "Then: chromapi flash, chromapi health, and https://$(hostname).local:5000"
echo "(or https://chromapi.local:5000 when it is the only robot on the network)"
