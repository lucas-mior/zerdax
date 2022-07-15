import numpy as np
class Angles:
    def set(angles):
        a0 = np.array([
              [1, 1, 1,],
              [3, 3, 3,],
              [1, 1, 1,],
              ], dtype='uint8').T
        a2 = np.array([
              [1, 1, 3,],
              [3, 3, 3,],
              [3, 1, 1,],
              ], dtype='uint8').T
        a4 = np.array([
              [1, 1, 3,],
              [1, 3, 1,],
              [3, 1, 1,],
              ], dtype='uint8').T
        a6 = np.array([
              [1, 3, 3,],
              [1, 3, 1,],
              [3, 3, 1,],
              ], dtype='uint8').T
        a9 = np.array([
              [1, 3, 1,],
              [1, 3, 1,],
              [1, 3, 1,],
              ], dtype='uint8').T

        i = 0
        k0 = np.zeros((angles.shape[0], 3, 3), dtype='uint8')
        k = np.zeros((angles.shape[0], 3, 3), dtype='uint8')
        for angle in angles:
            if abs(angle - 0.0) <= 11.25:
                k0[i,:,:] = a0
            elif abs(angle - 22.5) <= 11.25:
                k0[i,:,:] = a2
            elif abs(angle - 45.0) <= 11.25:
                k0[i,:,:] = a4
            elif abs(angle - 67.5) <= 11.25:
                k0[i,:,:] = a6
            elif abs(angle - 90) <= 11.25:
                k0[i,:,:] = a9
            else:
                k0[i,:,:] = a9
            i += 1

        i -= 1
        k = k0.sum(axis=0)
        k = np.uint8(k)
        return k
