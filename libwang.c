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
    for (int x = 1; x < xx-1; x++) {
        for (int y = 1; y < yy-1; y++) {
            W[yy*x + y] = weight(F, x, y, xx, yy);
        }
    }
    return;
}

void norm_array(double *F, int xx, int yy, double *W, double *N) {
    for (int x = 1; x < xx - 1; x++) {
        for (int y = 1; y < yy - 1; y++) {
            N[yy*x + y] = 0;
            for (int i = -1; i <= -1; i++) {
                for (int j = -1; j <= -1; j++) {
                    N[yy*x + y] += W[yy*(x+i) + y+j];
                }
            }
        }
    }
}
void convolute(double *F, int xx, int yy, double *W, double *N, double *filt) {
    for (int x = 1; x < xx - 1; x++) {
        for (int y = 1; y < yy - 1; y++) {
            filt[yy*x + y] = 0;
            for (int i = -1; i <= +1; i++) {
                for (int j = -1; j <= +1; j++) {
                    filt[yy*x + y] += (W[yy*(x+i) + y+j]*F[yy*(x+i) + y+j])/N[yy*x + y];
                }
            }
        }
    }
}

void wang_filter(double *F, int xx, int yy, double *W, double *N, double *filt) {
    weight_array(F, xx, yy, W);
    norm_array(F, xx, yy, W, N);
    convolute(F, xx, yy, W, N, filt);
}
