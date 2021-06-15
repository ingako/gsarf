#!/usr/bin/env bash
awk '{print $1" "$2" "$1+$2}' log.gsarf.time | awk '{ total += $3 } END { print total/NR }' | awk '{ print int($1/60)"m" ; print $1/60.0%1 * 60"s" }'
