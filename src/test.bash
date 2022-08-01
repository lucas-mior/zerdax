#!/bin/bash

for file in fotos/b*.jpg; do
    echo "zerdax $file"
    ./zerdax.py "$file" | sed -En '/――――/,/――――/p' > "${file/.jpg/.txt}"
done
