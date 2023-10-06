from skimage import transform

class Resize(object):
    
    '''
    Resize an image (or an image in a sample dictionary) to a given size.

    Args:
        output_size (tuple): contains ints with desired output size
        key (string):        specifies the image keyname in the dict
        
    Inputs:
        sample (ndarray or dict): sample containing input image and timestamp
        
    Returns:
        sample (ndarray or dict): sample with resized image
    '''

    def __init__(self, output_size, key=None):
        
        self.output_size = output_size
        self.key = key
        
        assert isinstance(output_size, (list, tuple))

    def __call__(self, sample):
        
        # resize
        if self.key is None:
            sample = transform.resize(sample, self.output_size)
        else:
            sample[self.key] = transform.resize(sample[self.key], 
                                                self.output_size)
        
        return sample