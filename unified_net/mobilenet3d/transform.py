import numpy as np

class ShapeTransform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):

        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()

        return voxel.astype(np.float32)