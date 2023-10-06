import torchvision

def Compose(transforms):
    
    '''
    Composes list of transformations into a sequence of operations.
    Note that these are performed in the order of the list.
    '''
    
    return torchvision.transforms.Compose(transforms)