#!/bin/bash
# Wrapper script for run_client.py that automatically restarts the client if it
# terminates.
# It keeps track of the number of restarts and aborts if there are too many
# within a certain time window (see variables below).
# Comprl server information can either be set as environment variables below or
# be passed as arguments to this script (any arguments will simply be forwarded
# to run_client.py).
#
# NOTE: If you want to stop the script, you need to press Ctrl+C twice.  Once
# to stop the client and then again to stop the wrapper script.

# Set defaults:
THRESHOLD_TIME_WINDOW=600
THRESHOLD=10

FILE_DIR="$(dirname "$(realpath "$0")")"

# load .env file
if [ -f $FILE_DIR/.env ]; then
    while read -r line_i || [ -n "$line_i" ]; do
        if [[ ! "$line_i" =~ ^#.*$ && "$line_i" =~ .*=.* ]]; then
            export "$line_i"
        fi
    done < $FILE_DIR/.env
fi
if [[ -z "${COMPRL_SERVER_URL}" || -z "${COMPRL_SERVER_PORT}" || -z "${COMPRL_ACCESS_TOKEN}" ]]; then
    echo "Error: COMPRL_SERVER_URL, COMPRL_SERVER_PORT and COMPRL_ACCESS_TOKEN must be set as environment variables or in a .env file."
    exit 1
fi

# get logfile name from environment variable or use default
LOGFILE="${LOGFILE:-client.log}"

log () {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] ${message}" >> "${LOGFILE}"
}

notify () {
    local message="$1"
    if [[ -n "${TG_BOT_TOKEN}" && -n "${TG_CHAT_ID}" ]]; then
        # telegram notification : export TG_BOT_TOKEN, TG_CHAT_ID
        local bot_token="${TG_BOT_TOKEN}"
        local chat_id="${TG_CHAT_ID}"

        curl -s -X POST "https://api.telegram.org/bot${bot_token}/sendMessage" \
            -d chat_id="${chat_id}" \
            -d text="${message}" > /dev/null
    fi
    if [[ -n "${NTFY_TOPIC}" ]]; then
        # ntfy notification : export NTFY_TOPIC
        local topic="${NTFY_TOPIC}"
        curl -s -X POST "https://ntfy.sh/${topic}" -d "${message}" > /dev/null
    fi

    log "Notification: ${message}"
}


# Array to hold timestamps of terminations
termination_times=()

while true; do
    # Run the command foobar
    python3 -m src.TDMPC.comprl_tdmpc

    # Get the current timestamp
    current_time=$(date +%s)

    # Add the current timestamp to the termination times array
    termination_times+=("$current_time")

    # Remove timestamps outside of the time window
    termination_times=($(for time in "${termination_times[@]}"; do
        if (( current_time - time <= ${THRESHOLD_TIME_WINDOW} )); then
            echo "$time"
        fi
    done))

    # Check if the number of terminations exceeds the threshold
    if (( ${#termination_times[@]} > ${THRESHOLD} )); then
        notify "ALERT: Client restarted too many times within the last ${THRESHOLD_TIME_WINDOW} seconds. Aborting."
        exit 1
    fi

    # Wait a bit before restarting.  In case the client terminated due to a
    # server restart, it will take some time before the server is ready again.
    sleep 20
    notify "Client restarted!"
done
