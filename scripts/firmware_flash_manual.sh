#!/usr/bin/env bash
# firmware_flash_manual.sh - Manually flash the STM32G431KBT6 firmware on the Chromapi motherboard
#
# Wiring (see https://github.com/Mowibox/chromapi_motherboard/blob/main/hardware/schematic_pdf/):
#   RPi UART_TXD (Pin 8)  -> STM32 USART1_RX (PA10)
#   RPi UART_RXD (Pin 10) -> STM32 USART1_TX (PA9)
#   RPi GPIO22   (Pin 15) -> STM32 BOOT0 (PB8)
#   RPi GPIO23   (Pin 16) -> STM32 NRST (PG10)
#
# Requirements:
#   sudo apt install stm32flash gpiod
#
# Usage:
#   ./firmware_flash_manual.sh <path/to/firmware.hex>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="/dev/ttyAMA0"
BAUD="115200"

GPIOCHIP="gpiochip0"
LINE_BOOT0=22
LINE_NRST=23
INIT_RETRIES=10

log() { echo "[firmware_flash_manual] $*"; }
fail() { echo "[firmware_flash_manual][FATAL] $*" >&2; exit 1; }

# Argument validation
if [ "$#" -ne 1 ]; then
    fail "Usage: $0 <path/to/firmware.hex>"
fi

HEX_FILE="$1"

command -v stm32flash >/dev/null 2>&1 || fail "stm32flash not found. Install with: sudo apt install stm32flash"
command -v gpioset     >/dev/null 2>&1 || fail "gpioset not found. Install with: sudo apt install gpiod"
command -v gpioget     >/dev/null 2>&1 || fail "gpioget not found. Install with: sudo apt install gpiod"
[ -r "$HEX_FILE" ]                     || fail "Hex file not found or not readable: $HEX_FILE"
[ -e "$PORT" ]                         || fail "$PORT does not exist."
[ -e "/dev/${GPIOCHIP}" ]              || fail "/dev/${GPIOCHIP} does not exist. Run 'gpiodetect' and fix GPIOCHIP in this script."

GPIOD_VERSION="$(gpioset --version 2>&1 | head -n1 || true)"
log "Detected: ${GPIOD_VERSION}"

set_pin() {
    local line="$1" value="$2"
    gpioset --mode=exit "$GPIOCHIP" "${line}=${value}"
}

get_pin() {
    local line="$1"
    gpioget "$GPIOCHIP" "$line"
}

enter_bootloader() {
    log "Entering STM32 system bootloader (BOOT0=high, pulse NRST)..."
    set_pin "$LINE_BOOT0" 1   # BOOT0 -> high
    set_pin "$LINE_NRST" 0    # hold in reset
    sleep 0.1
    set_pin "$LINE_NRST" 1    # release reset; BOOT0 sampled high by the ROM bootloader
    sleep 0.3                 # let the bootloader init USART1 / autobaud detect
}

exit_bootloader() {
    log "Restoring BOOT0=low and hardware-resetting into user flash..."
    set_pin "$LINE_BOOT0" 0
    set_pin "$LINE_NRST" 0
    sleep 0.1
    set_pin "$LINE_NRST" 1
}

# Always try to leave the board in a bootable state, even on failure.
trap exit_bootloader EXIT

enter_bootloader

log "Flashing $HEX_FILE on $PORT at ${BAUD} baud..."
if ! stm32flash -b "$BAUD" -w "$HEX_FILE" -v -n "$INIT_RETRIES" "$PORT"; then
    fail "stm32flash reported an error. Board left held in reset by the EXIT trap — check /boot/firmware/config.txt UART overlay and wiring before retrying."
fi

log "Flash + verify completed successfully."