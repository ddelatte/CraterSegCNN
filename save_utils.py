import pickle
import datetime


def save_hist(hist, filename='filename', ADD_TIME=True):
    now = ''
    if(ADD_TIME):
        now = datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    fn = filename + now + '.pickle'
    with open(fn, 'wb') as handle:
        #hist.history is passed in, so this is a "dict" type
        #TODO: check type
        if(type(hist)==dict):
            pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    with open(fn, 'rb') as handle:
        test = pickle.load(handle)
        
    print('File pickled? ' + str(hist == test))
    print(fn)
    
    return fn
