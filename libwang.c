#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double weight(double *F, int x, int y, int xx, int yy) {
    double Gx, Gy;
    double d, w;

    Gx = (F[yy*(x+1) + y] - F[yy*(x-1) + y]) / 2;
    Gy = (F[yy*x + y+1] - F[yy*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void weight_array(double *F, int xx, int yy, double *W) {
    printf("weight_array(%p, %d, %d)\n", F, xx, yy);

    for (int x = 1; x < xx-1; x++) {
        for (int y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(F, x, y, xx, yy);
        }
    }
    printf("C: W[%d*200 + 200] = %f\n", xx, W[yy*200 + 150]);
    return;
}
