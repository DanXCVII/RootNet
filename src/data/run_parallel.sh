#!/bin/bash

run_script() {
    while true; do
        python3 generator.py
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo -e "\e[31m \nScript exited with error $exit_status, restarting...\e[0m"
        else
            break
        fi
    done
}

for i in {1..2}; do
    run_script &
done

wait