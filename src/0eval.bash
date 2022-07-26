#!/bin/bash

for result in *corners.png; do
    nsxiv "$result" > /tmp/00eval
    if grep -q "^1" /tmp/00eval; then 
        mv -t certo/ "${result/corners.png/}"*
    else
        mv -t errad/ "${result/corners.png/}"*
    fi
done
