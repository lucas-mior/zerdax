#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double weight(double *F, int x, int y, int xx, int yy) {
    double Gx, Gy;
    double d, w;

    Gx = (F[xx*(x+1) + y] - F[xx*(x-1) + y]) / 2;
    Gy = (F[xx*x + y+1] - F[xx*x + y-1]) / 2;

    d = sqrt(Gx*Gx + Gy*Gy);
    w = exp(-sqrt(d));
    return w;
}

void weight_array(double *F, int xx, int yy, double *W) {
    printf("weight_array(%p, %d, %d)\n", F, xx, yy);

    for (int x = 1; x < xx-1; x++) {
        for (int y = 1; y < yy-1; y++) {
            W[xx*x + y] = weight(F, x, y, xx, yy);
        }
    }
    printf("C: W[%d*200 + 200] = %f\n", xx, W[xx*200 + 200]);
    return;
}
