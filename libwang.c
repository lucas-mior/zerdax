#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double weight(double **f, int x, int y) {
    double Gx, Gy;
    double d, w;

    Gx = (f[x+1][y] - f[x-1][y]) / 2;
    Gy = (f[x][y+1] - f[x][y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

double *weight_array(double **f, int xx, int yy) {
    printf("weight_array(%p, %d, %d)\n", f, xx, yy);
    double *W;

    if (W = malloc(sizeof(double) * (xx) * (yy))) {
        printf("W = %p\n",W);
    } else {
        printf("malloc failed\n");
        exit(1);
    }

    for (int x = 1; x < xx-1; x++) {
        for (int y = 1; y < yy-1; y++) {
            W[xx*x + y] = weight(f, x, y);
        }
    }
    printf("C: W[%d*20 + 1] = %f", xx, W[xx*20 + 1]);

    return W;
}
