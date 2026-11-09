#!/usr/bin/env bash
# chromapi.local, besides the robot's own chromapi-xxxx.local (chromapi-alias.service). Installed as
# /usr/local/sbin/chromapi-alias. One Chromapi on the network (a home, the assembly bench) would always answers at
# https://chromapi.local:5000, whatever its name. Several: the first one switched on keeps it (the others get
# "Local name collision" and try again later); "Identify" button in the page tells which robot it is.
ALIAS=chromapi.local
RETRY_S=30  # [s] after a collision, or no network
pid="" ip=""
while true; do
    now="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then pid=""; fi  # avahi-publish gave up (collision)
    if [ "$now" != "$ip" ] || [ -z "$pid" ]; then
        [ -z "$pid" ] || kill "$pid" 2>/dev/null
        pid="" ip="$now"
        if [ -n "$ip" ]; then
            avahi-publish -a -R "$ALIAS" "$ip" &  # -R: no reverse entry, the IP already maps to chromapi-xxxx
            pid=$!
        fi
        sleep "$RETRY_S"
    else
        sleep 5  # follow a new IP (another network) within seconds
    fi
done
