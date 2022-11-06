#!/bin/bash

overall=0
for status_file in /proc/[0-9]*/status; do
    swap_mem=$(grep VmSwap "$status_file" | awk '{ print $2 }')
    if [ "$swap_mem" ] && [ "$swap_mem" -gt 0 ]; then
        pid=$(grep Tgid "$status_file" | awk '{ print $2 }')
        name=$(grep Name "$status_file" | awk '{ print $2 }')
        printf "%s\t%s\t%s KB\n" "$pid" "$name" "$swap_mem"
    fi
    overall=$((overall+swap_mem))
done
printf "Total Swapped Memory: %14u KB\n" $overall

