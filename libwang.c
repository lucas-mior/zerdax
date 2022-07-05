#include <stdint.h>
#include <stdio.h>
#include <math.h>

double weight(double **f, int x, int y) {
    float Gx, Gy;
    float d, w;

    Gx = (f[x+1][y] - f[x-1][y]) / 2;
    Gy = (f[x][y+1] - f[x][y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}
