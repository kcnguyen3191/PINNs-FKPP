import numpy as np

class Flip(object):
    
    '''
    Randomly flips a square image in a sample left/right, up/down, or both.

    Args:
        lr    (bool): optional left/right flip
        ud    (bool): optional up/down flip
        key (string): specifies the image keyname in the dict
        
    Inputs:
        sample (ndarray or dict): sample containing input image with shape (C, H, W)
        
    Returns:
        sample (ndarray or dict): sample with rotated image with shape (C, H, W)
    '''
    
    def __init__(self, lr=True, ud=False, key=None):
        
        self.lr = lr
        self.ud = ud
        self.key = key

    def __call__(self, sample):
        
        # random flips
        lr = np.random.choice([True, False])
        ud = np.random.choice([True, False])
        
        # if sample is the image itself
        if self.key is None:
            
            # loop over channels
            for i in range(sample.shape[0]):
                
                # left/right flip
                if self.lr and lr:
                    sample[i] = np.fliplr(sample[i])
                    
                # up/down flip
                if self.ud and ud:
                    sample[i] = np.flipud(sample[i])
                    
            # copy resets ndarray strides to make torch happy
            sample = sample.copy()
        
        # otherwise dictionary
        else:
            
            # loop over channels
            for i in range(sample[self.key].shape[0]):
                
                # left/right flip
                if self.lr and lr:
                    sample[self.key][i] = np.fliplr(sample[self.key][i])
                    
                # up/down flip
                if self.ud and ud:
                    sample[self.key][i] = np.flipud(sample[self.key][i])
                    
            # copy resets ndarray strides to make torch happy
            sample[self.key] = sample[self.key].copy()

        return sample