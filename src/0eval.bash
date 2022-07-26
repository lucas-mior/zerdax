#!/bin/bash

for result in *corners.png; do
    nsxiv "$result" > /tmp/00eval
    if grep -q "^1" /tmp/00eval; then 
        mv -t certo/ ${result%%corners*}*
    else
        mv -t errad/ ${result%%corners*}*
    fi
done
