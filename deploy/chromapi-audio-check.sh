#!/usr/bin/env bash
# Before the speaker keep-alive, as root. Installed as /usr/local/sbin/chromapi-audio-check.
# Ihe sound card comes up broken at boot: this script re-bind the codec,then the card.
CARD=sndrpigooglevoi
DRIVERS=/sys/bus/platform/drivers

try_open() {  # 0.1 s of silence on the bare card; prints aplay's error
    aplay -q -D "hw:$CARD,0" -t raw -f S32_LE -r 48000 -c 2 -s 4800 /dev/zero 2>&1 >/dev/null
}

for _ in $(seq 20); do grep -q "\[$CARD" /proc/asound/cards && break; sleep 0.5; done  # the card, up to 10 s

error="$(try_open)"
[[ "$error" == *"Invalid argument"* ]] || exit 0  # fine, busy (the app already plays) or absent: not ours to mend

echo "Sound card broken at boot ($error): re-binding the codec and the card"
echo soc:sound >"$DRIVERS/snd-rpi-simple/unbind" 2>/dev/null
echo voicehat-codec >"$DRIVERS/voicehat-codec/unbind"
echo voicehat-codec >"$DRIVERS/voicehat-codec/bind"
echo soc:sound >"$DRIVERS/snd-rpi-simple/bind" 2>/dev/null  # "busy": already bound again with the codec
for _ in $(seq 20); do grep -q "\[$CARD" /proc/asound/cards && break; sleep 0.5; done
error="$(try_open)"
if [ -z "$error" ]; then echo "Sound card mended"; else echo "Sound card still broken ($error): reboot" >&2; fi
exit 0  # the keep-alive starts anyway: its own retries are the fallback
