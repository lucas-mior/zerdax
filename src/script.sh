#!/bin/sh

for a in $(seq 20 1 20); do
    for b in $(seq 120 1 120); do
        for c in $(seq 20 20 100); do
            for d in $(seq 90 30 300); do
                for e in $(seq 6 6 24); do
                    # printf "a = $a"
                    # printf "b = $b"
                    # printf "c = $c"
                    # printf "d = $d"
                    # echo "e = $e"
                    ./zerdax.py "$1" -a $a -b $b -c $c -d $d -e $e
                done
            done
        done
    done
done
