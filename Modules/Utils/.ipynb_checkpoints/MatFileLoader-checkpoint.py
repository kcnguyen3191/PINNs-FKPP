import scipy.io, h5py

def MatFileLoader(path_to_file):
    
    '''
    Loads .mat files of version 7.3 and earlier.
    '''
    
    # try to use scipy
    try:
        file = scipy.io.loadmat(path_to_file)
        
    # except if the .mat file is from version 7.3
    except:
        with h5py.File(path_to_file, 'r') as f:
            file = f
        #raise NotImplementedError('Not implemented yet.')
        
    return file