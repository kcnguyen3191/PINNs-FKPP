import numpy as np

class Rotate90(object):
    
    '''
    Randomly rotates a square image in a sample by 0, 90, 180, or 270 degrees
    with optional random transposing.

    Args:
        transpose (bool): optional transpose in addition to rotations
        key (string):     specifies the image keyname in the dict
        
    Inputs:
        sample (ndarray or dict): sample containing input image with shape (C, H, W)
        
    Returns:
        sample (ndarray or dict): sample with rotated image with shape (C, H, W)
    '''
    
    def __init__(self, transpose=False, key=None):
        
        self.transpose = transpose
        self.key = key

    def __call__(self, sample):
        
        # random number of rotations and random transpose
        r = np.random.choice([0, 1, 2, 3])
        t = np.random.choice([True, False])
        
        # if sample is the image itself
        if self.key is None:
            # loop over channels
            for i in range(sample.shape[0]):
                sample[i] = np.rot90(sample[i], r)
                if self.transpose and t:
                    sample[i] = sample[i].T
            # copy resets ndarray strides to make torch happy
            sample = sample.copy()
        
        # otherwise dictionary
        else:
            # loop over channels
            for i in range(sample[self.key].shape[0]):
                sample[self.key][i] = np.rot90(sample[self.key][i], r)
                if self.transpose and t:
                    sample[self.key][i] = sample[self.key][i].T
            # copy resets ndarray strides to make torch happy
            sample[self.key] = sample[self.key].copy()

        return sample