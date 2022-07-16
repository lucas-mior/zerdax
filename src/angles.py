import numpy as np
def set_kernels(angles):
    a0 = np.array([
          [0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0],
          ], dtype='float32').T
    a2 = np.array([
          [0, 0, 0, 0, 0],
          [0, 0, 0, 1, 2],
          [0, 1, 2, 1, 0],
          [1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0],
          ], dtype='float32').T
    a4 = np.array([
          [0, 0, 0, 1, 2],
          [0, 0, 1, 2, 1],
          [0, 1, 2, 1, 0],
          [1, 2, 1, 0, 0],
          [2, 1, 0, 0, 0],
          ], dtype='float32').T
    a6 = np.array([
          [0, 0, 0, 2, 0],
          [0, 0, 1, 1, 0],
          [0, 0, 2, 0, 0],
          [0, 1, 1, 0, 0],
          [0, 2, 0, 0, 0],
          ], dtype='float32').T
    a9 = np.array([
          [0, 1, 2, 1, 0],
          [0, 1, 2, 1, 0],
          [0, 1, 2, 1, 0],
          [0, 1, 2, 1, 0],
          [0, 1, 2, 1, 0],
          ], dtype='float32').T

    k0 = np.zeros((angles.shape[0], 5, 5), dtype='float32')
    k = np.zeros((angles.shape[0], 5, 5), dtype='float32')
    i = 0
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
    # k = k0.sum(axis=0)
    # k = np.float32(k)
    return k0
