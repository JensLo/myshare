# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:21:48 2013

@author: LRJ1si
"""
import numpy as np
import pylab as plt
from matplotlib.axes import Subplot
from funcs import *
from ETKData import devicelist, dotdict
import matplotlib.lines as mlines
import ETKPlot as eplt

from collections import MutableSequence
        
def slice_equal(x,y):
    x = x[:len(y)]
    y = y[:len(x)]
    return (x,y)

class eventlist(object): 
    
    def __getitem__(self, k):
        return self.__dict__[k]
        
    def __setitem__(self, k, v):
        self.__dict__[k] = v
        
    def keys(self):
        return self.__dict__.keys()
        
    def values(self):
        return self.__dict__.values()
        
    def items(self):
        return self.__dict__.items()
        
    def __delitem__(self, k):
        """Delete an item"""
        del self.__dict__[k]

    def __repr__(self):
        ret = 'Channels:\n'
        for channelname, channel in sorted(self.items()):
            ret += '  ' + str(channelname) + ':\n'
            for k,v in sorted(channel.info.items()):
               ret += '    ' + k +': ' + v + '\n'

        return ret
        
class event(object):

    def __init__(self, channel, idxs):
        self.__channel__ = channel
        self.__idxs__ = idxs
        for ix in range(len(idxs[0])):
            self.__dict__['ix'+str(ix)] = self[ix]
        for key, info in self.__channel__.info.items():
            self.__dict__[key] = info
        self.info = self.__channel__.info

    def __len__(self):
        return len(self.__idxs__[0])

    def __getitem__(self, k):
        try:
            int(k)
            return dotdict(x=self.__channel__.x[self.__idxs__ [0][k]:self.__idxs__ [1][k]],
                    y=self.__channel__.y[self.__idxs__ [0][k]:self.__idxs__ [1][k]],
                    time=self.__channel__.x[self.__idxs__ [0][k]:self.__idxs__ [1][k]],
                    data=self.__channel__.y[self.__idxs__ [0][k]:self.__idxs__ [1][k]])
        except:
            try:
                return self.__channel__[k]
            except:
                raise KeyError('list indices must be integers or slices, not str')

    def __repr__(self):
        _idxs0 = self.__idxs__[0]
        _idxs1 = self.__idxs__[1]
        ret = self.__channel__.__repr__() 
        for ix in range(len(_idxs0)):
            ret += 'Event ' + str(ix) + ': [' 
            ret +=  'index='+str(_idxs0[ix]) + '@time=%.1fs, '%(self.__channel__.x[_idxs0[ix]])
            ret +=  'index='+str(_idxs1[ix]) + '@time=%.1fs]\n'%(self.__channel__.x[_idxs1[ix]])

        return ret

    def pop(self, idx):
        self.__idxs__[0].pop(idx)
        self.__idxs__[1].pop(idx)
        


    def plot(self, fnc_x=lambda x: x, fnc_y=lambda y:y, ax=None, f=None, xlabel=None, ylabel=None, 
             label=None, ls='-', color=None, drawstyle='steps', lib='matplotlib', shift=False,
             idxs=None, **figargs):

        idxs = np.arange(len(self)) if idxs is None else idxs

        if ax is None:
            xlabel = r'time / s'
            ylabel = self.name + ' / ' + self.unit
            label = self.longname

        xlabel = r'time / s' if xlabel is None else xlabel
        ylabel = self.name + ' / ' + self.unit if ylabel is None else ylabel
        label = None ## should be set from outside
        
        eplt.set_plottinglib(lib)
        f, ax = eplt.make_figure(1,1, ax, **figargs)
        
        for ix in range(len(self)):
            if (ix in idxs):
                time = fnc_x(self[ix].x)
                data = fnc_y(self[ix].y)
                time = time - time[0] if shift else time

                l = eplt.doplot(time, data, ax[0], label, xlabel, ylabel, 
                                ls, drawstyle, color, name=self.name)
                xlabel = None
                ylabel = None
                legend = False

        return f, ax


class ETKEvents(object):
        
    def __init__(self, data):
        self.__data__ = data
        self.__D__ = devicelist()
        self.names = dotdict()
        
    def add( self, channel, idxs):
       self[channel.device+'/'+channel.name] = event(channel, idxs)
       self.names[channel.name+ '_' + channel.device + '_'] = channel.device+'/'+channel.name


    def __getitem__(self, key):
        if isinstance(key, eventlist):
            key = list(key.values())[0].device
        if isinstance(key, event):
            key = key.device + '/' + key.name

        if key in self.keys():
            return self.__D__[key]


        if key.count('/')==0:
            _key = self.__data__._getDeviceChannelName(key)
            if _key is None:
                raise KeyError('Channel ' + str(key) + ' not found')
                return key
            else:
                key = _key

        key = key.split('/')
        try:
            return self.__D__[key[0]][key[1]]
        except:
            if key[0] not in self.keys():
                self.__setitem__(key[0], eventlist())

            return self.__D__[key[0]][key[1]] 

            
    def __setitem__(self, key, value):
        if type(value) == type(eventlist()):
            self.__D__[key] = value
            self.__dict__[key] = self.__D__[key]
            return
        if key in self.keys():
            self.__D__[key] = value
            return
        if key.count('/')==0:
            _key = self.__data__._getDeviceChannelName(key)
            if _key is None:
                raise KeyError('Channel ' + str(key) + ' not found')
                return key
            else:
                key = _key

        key = key.split('/')
        try:
            self.__D__[key[0]][key[1]] = value
        except:
            if key[0] not in self.keys():
                self.__D__[key[0]] = eventlist()
                self.__dict__[key[0]] = self.__D__[key[0]]
            self.__D__[key[0]][key[1]] = value
        
    def keys(self):
        return self.__D__.keys()

    def values(self):
        return self.__D__.values()

    def items(self):
        return self.__D__.items()
        

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __repr__(self):
        pass
        """ret = 'Channels:\n'
                                for channelname, channel in sorted(self.items()):
                                    ret += '  ' + str(channelname) + ':\n'
                                    for k,v in sorted(channel.info.items()):
                                       ret += '    ' + k +': ' + v + '\n'
                        
                                return ret"""
        
        
        
    def remove(self, filt_func, *args, **kwargs):
        idxs = []  
        _args = []
        for a in np.atleast_1d(args):
            _args.append(self[a])
            

        for i in range( len(_args[0]) ):
            if filt_func(i, *_args, **kwargs):
                idxs.append(i)
                    
        for dev in self.__D__.keys():
            for k in self.__D__[dev].keys():
                for i in idxs[::-1]:
                    self.__D__[dev][k].pop(i)
            

    def plot(self, y, kind='', ax=0, fnc_x = lambda x:x, fnc_y = lambda y:y, 
             row_c = 1, col_c = 1, start_idx = None, stop_idx = None, step = 1,
             filt = (None, None), shift_time = False, label=True, legend=True,
             drawstyle='steps', **figkw):
        
        x_size, y_size = plt.rcParams['figure.figsize']
        
        if str(type(y)) == "<class 'ETKResult.result_list'>":
            y = [[y]]
        elif type(y) in (tuple, list, np.ndarray):
            if np.any([str(type(e)) == "<class 'ETKResult.result_list'>" for e in y]):
                y = [[e] for e in y if str(type(e)) == "<class 'ETKResult.result_list'>" ]
        else:
            y = np.atleast_1d(y)

        if row_c == 1:
            row_c = len(y)
                 
        if type(np.atleast_1d(ax)[0]) != Subplot:
            if row_c > 0 and col_c > 0:
               axs = mk_subplot( row_c, col_c, figsize = (x_size, row_c * 0.5 * y_size)  )
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
                
                if type(yy) == np.str_:
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
                        y_vals = fnc_y( yy[i].y )
                        time = (fnc_x( yy[i].x ) - \
                             (fnc_x( yy[i].x )[0] if shift_time else 0.0))[:y_vals.shape[0]]
                        y_vals = y_vals[:time.shape[0]]
                        
                        ax.plot( time, y_vals, label = yy[i].longname, 
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
            return zip( *( [self[k][i] for i in range(len(self[k]))] for k in keys  ) )
            
    def findKeys(self, filter = '.*'):
        import re
        ret = []
        filter = np.atleast_1d(filter)
        for dk in self.D.keys():
            for ck in self.D[dk].keys():
                for kf in filter:
                    if re.match(kf, ck) is not None:
                        ret.append(dk+'/'+ck)
                        break
                
        return ret
