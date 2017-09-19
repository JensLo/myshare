# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:21:48 2013

@author: LRJ1si
"""

import numpy as np

def sliceat(data, idxs):
    tmp = []
    for idx1, idx2 in idxs:
        tmp.append(data[idx1:idx2])
        
    return np.array(tmp)

#   try:
#        tmp = idxs[ np.diff(idxs)>1 ]
#        ret = tmp[((nEng.argmax() - tmp)).argmin()]
#    except:
#        ret = idxs[0]
def findFilterIdxs(data, cons):
    filt = None
    for key, filt_func in cons:
        
        if filt is None:
            filt = filt_func(data[key].y)
        else:
            filt = ( filt == filt_func(data[key].y) )
        
    return np.arange(data[key].y.shape[0])[filt]
    

def filterData(data, filterIdxs):
    return data[filterIdxs]





