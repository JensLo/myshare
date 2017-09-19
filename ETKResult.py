# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:21:48 2013

@author: LRJ1si
"""
import numpy as np
import pylab as plt
from matplotlib.axes import Subplot
from funcs import *
from ETKData import device, data, channel
import matplotlib.lines as mlines

        
class Results(object):
        
    def __init__(self, def_device='ETKC_1'):
        self.D = device()
        self._def_device=def_device
        
    def add( self, key, chan, idxs, samplingTime=None ):
        from scipy.interpolate import interp1d
        dev, name = key.split('/')
        if samplingTime is not None:
            f = interp1d( chan.x, chan.y, kind ='zero', bounds_error=False )
            dat = f(samplingTime)
            time = samplingTime
        else:
            dat = chan.y
            time = chan.x
         
        for idx1, idx2 in zip(idxs[0], idxs[1]):
            self[key].append( channel(dat[idx1:idx2], 
                                      time[idx1:idx2], 
                                      time = chan.time,
                                      description =chan.description,
                                      unit = chan.unit,
                                      name = name,
                                      longname = name + '(' + dev + ')',
                                      device = dev) )
                                          
        self.__dict__['_k_'+name+ '_' + dev + '_'] = dev + '/' + name
        
    def calc( self, key, fnc = lambda x:x, *args, **kwargs):
                    
        #globals()['%s__%s'%(self.__name__, name)] = name
        t_time = ''
        _args = []
        
        for a in np.atleast_1d(args):
            try:
                _args.append(self[a])
                time = self[a].x
                t_time = self[a].time
            except:
                _args.append(a)

        
        #self[key] = []
        dev, name = ('Calc',key)
        chan = _args[0]
        for all_a in np.array(_args).T:
            
            self[dev + '/' + name].append( channel(fnc[0](*all_a),
                                      chan.x, 
                                      time = chan.time,
                                      unit = kwargs.pop('unit', r'a.u.'), 
                                      description = kwargs.pop('description', r'calculated ...'), 
                                      name = name,
                                      longname = name + '(' + dev + ')',
                                      device = dev) )
            
        self.__dict__['_k_'+name+ '_' + dev + '_'] = dev + '/' + name
        
        
        
    def __getitem__(self, key):
        if np.isscalar(key):
            if key.count('/')>0:
                key = key.split('/')
                try:
                    return self.D[key[0]][key[1]]
                except:
                    if key[0] not in self.D.keys():
                        self.D[key[0]] = data()
                    self.D[key[0]][key[1]] = []
                    return self.D[key[0]][key[1]]
            else:
                if key not in self.D.keys():
                    return self.D[self._def_device][key]
                else:
                    self.D[key] = data()
                    return self.D[key]
        else:
            return key
            
    def __setitem__(self, key, value):
        if np.isscalar(key):
            if key.count('/')>0:
                key = key.split('/')
                try:
                    self.D[key[0]][key[1]] = value
                except:
                    if key[0] not in self.D.keys():
                        self.D[key[0]] = data()
                    self.D[key[0]][key[1]] = value
            else:
                self.D[key] = value

        else:
            key = value
        
    def remove(self, filt_func, *args):
        idxs = []  
        _args = []
        for a in np.atleast_1d(args):
            try:
                _args.append(self[a])
            except:
                _args.append(a)
                
        for i in range( len(_args[0]) ):
            if filt_func(i, *_args):
                idxs.append(i)
                    
        for dev in self.D.keys():
            for k in self.D[dev].keys():
                for i in idxs[::-1]:
                    self.D[dev][k].pop(i)
            

    def plot(self, y, kind='', ax=0, fnc_x = lambda x:x, fnc_y = lambda y:y, 
             row_c = 1, col_c = 1, start_idx = None, stop_idx = None, step = 1,
             filt = (None, None), shift_time = False, label=True, legend=True,
             drawstyle='steps', **figkw):
             
        if type(y) not in (list, tuple, np.ndarray):
            y = [y]
            
        if row_c == 1:
            row_c = len(y)
                 
        if type(np.atleast_1d(ax)[0]) != Subplot:
            if row_c > 0 and col_c > 0:
               axs = mk_subplot( row_c, col_c, **figkw )
            else:
                axs = [plt.figure(**figkw).add_subplot(111)]
        else:
            axs = np.atleast_1d(ax)

        for j, yy1 in enumerate(y): 
            y_label = ''
            j = min(j, len(axs)-1)
            ax = axs[j]
            print_label=label
            handles=[]
            for yy, ls in zip(np.atleast_1d(yy1), ('-','--','-.',':')):  
                yy = self[yy]
                start = start_idx if start_idx is not None else 0
                if stop_idx is not None:
                    if abs(stop_idx) != stop_idx:
                        stop_idx = len(yy) - stop_idx + 1
                stop = min(stop_idx+1, len(yy)) if stop_idx is not None else len(yy)
                
                for i in range(start, stop, step):
                    doplot = True
                    
                    if filt[0] is not None:
                        filt_func, filt_arg = filt
                        args = [self[ff][i].y for ff in np.atleast_1d(filt_arg)]
                        doplot = filt_func(*args)
                    
                    if doplot:
                        xax = fnc_x(yy[i].x ) - \
                             (fnc_x( yy[i].x )[0] if shift_time else 0.0)
         
                        ax.plot( xax, fnc_y( yy[i].y ), label = yy[i].longname, 
                                 linewidth=2, drawstyle=drawstyle, ls=ls )
                
                if print_label:
                    handles.append((mlines.Line2D([], [], color='k', ls=ls), yy[i].longname))
                
                ax.grid( True )
                try:
                    y_label += yy[0].unit + '; '
                except:
                    print(yy)
            
                
                try:
                    ax.set_ylabel( y_label[:-2] )
                except:
                    print('MOEP',y_label, yy[0].name)
                    
                ax.set_xlabel( r'time / s' )
               

                if kind == 'loglog':
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                elif kind == 'semilogy':
                    ax.set_xscale('linear')
                    ax.set_yscale('log')
                elif kind == 'semilogx':
                    ax.set_xscale('log')
                    ax.set_yscale('linear')
                    
                
            print_label=False
            if legend:
                ax.legend(*(np.array(handles).T.tolist()), ncol=len(np.atleast_1d(yy1)), fancybox=True, loc=0, frameon=True)
            num_lines = len(ax.get_lines())/len(np.atleast_1d(yy1))
            #fnc_c=color.color_mapper([0, num_lines-1], cmap)
            #if num_lines > 1:
            #    [l.set_color(fnc_c(i%num_lines)) for i,l in enumerate(ax.get_lines())]
            
                    
        plt.subplots_adjust(hspace = 0, wspace = 0)
        from matplotlib.ticker import MaxNLocator
        [ax.yaxis.set_major_locator(MaxNLocator(prune='upper')) for ax in axs]
        
        plt.draw()
        
        return axs
        
    def forKeys( self, *keys ):
        if len(keys) == 1:
            return self[keys[0]]
        else:
            return zip( *( self[k] for k in keys ) )
