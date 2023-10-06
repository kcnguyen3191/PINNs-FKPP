import numpy as np
import mxnet as mx

def channels_first(x):
    return np.swapaxes(np.swapaxes(x, 0, 2), 1, 2)

def channels_last(x):
    return np.swapaxes(np.swapaxes(x, 0, 2), 0, 1)

class Color(object):
    
    '''
    Randomly augments the colors of an image using .

    Args:
        brightness (float): max amount of change in brightness
        contrast   (float): max amount of change in contrast
        saturation (float): max amount of change in saturation
        hue        (float): max amount of change in hue
        clip        (list): min and max values to clip augmented image
        key       (string): specifies the image keyname if using dict as input
        
    Inputs:
        image (ndarray or dict): input image with shape (C, H, W)
        
    Returns:
        image (ndarray or dict): augmented image with shape (C, H, W)
    '''
    
    def __init__(self, 
                 brightness=0.0, 
                 contrast=0.0, 
                 saturation=0.0, 
                 hue=0.0,
                 clip=[0, 1],
                 key=None):
        
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.clip = clip
        self.key = key

    def __call__(self, image):
        
        # generate augmentation
        aug_list = [
            mx.image.BrightnessJitterAug(brightness=self.brightness),
            mx.image.ContrastJitterAug(contrast=self.contrast),
            mx.image.SaturationJitterAug(saturation=self.saturation),
            mx.image.HueJitterAug(hue=self.hue)
        ]
        aug = mx.image.RandomOrderAug(aug_list)
        
        # if sample is the image itself
        if self.key is None:
        
            # add random augmentations
            image = channels_last(image)
            image = mx.nd.array(image)
            image = aug(image).asnumpy().clip(*self.clip)
            image = channels_first(image)
        
        # otherwise dictionary
        else:
            
            # add random augmentations
            image[self.key] = channels_last(image[self.key])
            image[self.key] = mx.nd.array(image[self.key])
            image[self.key] = aug(image[self.key]).asnumpy().clip(*self.clip)
            image[self.key] = channels_first(image[self.key])

        return image