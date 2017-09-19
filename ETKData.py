# -*- coding: utf-8 -*-

import numpy as np

from mdfreader import mdf, mdfinfo
from funcs import *
import time
from rcParams_bosch import bosch_standard_palette
from rcParams_bosch import bosch_gray_palette

from difflib import SequenceMatcher

import ETKPlot as eplt

import os
import gc
import gzip
import dill



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class devicelist(object): 

    def __call__(self, k):
        return self.__dict__[k]

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

    def __repr__(self):
        ret = 'Devices:\n'
        for devicename, device in sorted(self.items()):
            ret += '  ' + devicename + ':\n'
            for k in sorted(device.keys()):
                ret += '    ' + k + '\n'

        return ret
    

class channellist(object): 
           
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
        
class channel(object):
    
    def __init__(self, cname, mdf, name, longname, device):
        self._cname = cname
        self.__mdf__ = mdf
        self.name = name
        self.longname = longname
        self.device = device
        self.info = dotdict(cname = cname,
                            name = name,
                            longname = longname,
                            device = device,
                            master = self.master,
                            unit = self.unit,
                            description = self.description)
        
    def __getattr__(self, attr):
        if attr in ('time', 'x'):
            return self.__mdf__.getChannelData(self.__mdf__[self._cname]['master'])
        if attr in ('data', 'y'):
            return self.__mdf__.getChannelData(self._cname)
        
        return self.__mdf__[self._cname][attr]
        
    def __getitem__(self, key):
        return self.__getattr__(key)
        
    def __setitem__(self, key, value):
        self[key] = value

    def __repr__(self):
        return  ''.join([k + '\t= ' + str(v) + '\n' for k,v in self.info.items()])

    def plot(self, fnc_x=lambda x: x, fnc_y=lambda y:y, ax=None, f=None, xlabel=None, ylabel=None, 
             label=None, ls='-', color=None, drawstyle='steps', lib='matplotlib',**figargs):

        if ax is None:
            xlabel = r'time / s'
            ylabel = self.name + ' / ' + self.unit
            label = self.longname

        xlabel = r'time / s' if xlabel is None else xlabel
        ylabel = self.name + ' / ' + self.unit if ylabel is None else ylabel
        label = None ## should be set from outside
        
        eplt.set_plottinglib(lib)
        f, ax = eplt.make_figure(1,1, ax, **figargs)

        time = fnc_x(self.x)
        data = fnc_y(self.y)

        l = eplt.doplot(time, data, ax[0], label, xlabel, ylabel, 
                        ls, drawstyle, color, name=self.name)

        return ax, l


class calcchannel(channel):

    def __init__(self, cname, mdf, name, longname, device):
        super(calcchannel, self).__init__(cname, mdf, name, longname, device)
        
    def __getattr__(self, attr):
        if attr in ('time', 'x'):
            return self.__mdf__[self.__mdf__[self._cname]['master']]['data']
        if attr in ('data', 'y'):
            return self.__mdf__[self._cname]['data']
        
        return self.__mdf__[self._cname][attr]
        
    

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __init__(self, **kwargs):
        super(dotdict, self).update(kwargs)
        self.__dict__.update(kwargs)

    def __getitem__(self, k):
        return self.__dict__[k]
        
    def __setitem__(self, k, v):
        super(dotdict, self).__setitem__(k,v)
        self.__dict__[k] = v
    
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
        

class ETKData(object):
    
    def __init__(self):
        self.__D__ = devicelist()
        self.__mdf__ = None
        self.names = dotdict()

       #self._min_max_times = [1e300, -1e300]
       

    def __getitem__(self, key):
        if isinstance(key, channellist):
            key = list(key.values())[0].device
        if isinstance(key, channel):
            key = key.device + '/' + key.name
        if key in self.keys():
            return self.__D__[key]
        if key.count('/')==0:
            _key = self._getDeviceChannelName(key)
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
                self.__setitem__(key[0], channellist())

            return self.__D__[key[0]][key[1]] 

            
    def __setitem__(self, key, value):
        if type(value) == type(channellist()):
            self.__D__[key] = value
            self.__dict__[key] = self.__D__[key]
            return
        if key in self.keys():
            self.__D__[key] = value
            return
        if key.count('/')==0:
            _key = self._getDeviceChannelName(key)
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
                self.__D__[key[0]] = channellist()
                self.__dict__[key[0]] = self.__D__[key[0]]
            self.__D__[key[0]][key[1]] = value

    def keys(self):
        return self.__D__.keys()

    def values(self):
        return self.__D__.values()

    def items(self):
        return self.__D__.items()

    
    def _getDeviceChannelName(self, chan):
        
        for dk in self.keys():
            if chan in  self[dk].keys():
                return dk+'/'+chan

        return None
            
     
    def _load(self,filterFunctionName):
        
        for chan in self.__mdf__.keys():
            if chan[0] not in ('%','$'):
                try:
                    ck, dk = chan.split('\\')
                    
                except:
                    continue

                if filterFunctionName:
                    ck = ck.split('.')[0]
                newDName = dk.replace(':','').replace(' ','').replace('-','_').replace('/','')
                newCName = ck.replace(':','').replace(' ','').replace('-','').replace('[','_ix').replace(']','').replace('/','_').replace('.','__')
                
                if not newDName in list(self.keys()):
                    self[newDName] = channellist()

                c_type = channel if isinstance(self.__mdf__, mdf) else calcchannel

                #cname, mdf, name, longname, device, 
                self[newDName][newCName] = c_type(cname = chan, 
                                                  mdf = self.__mdf__,
                                                  name = newCName,
                                                   longname = newCName + '(' + newDName + ')',
                                                   device = newDName)
                self.names[newCName+ '_' + newDName + '_'] = newDName + '/' + newCName

        gc.collect()


    def load(self, fname, resample = 0.00, cfilter = ['.*'], clist = None, 
             filterFunctionName=True, noDataLoading=True) :


        
        #if os.path.exists( fname ):
        self.filename = fname        
        self.filename = os.path.split(fname)[1],
        self.folder = os.path.split(fname)[0],
        self.date = time.strftime('%Y-%m-%d %H:%M:%S',
                                   time.localtime(os.path.getctime(fname)))

        try:
            self.__mdf__ = mdf( fname, cfilter, clist, noDataLoading=noDataLoading )
        except:
            import re
            with gzip.open(os.path.join(fname), 'rb') as pickle_file:
                self.__mdf__ = dill.load(pickle_file)
                keys_to_del = []
                for key in self.__mdf__.keys():
                    to_del = True
                    for kf in np.atleast_1d(cfilter):
                        if ((re.findall(kf, key) != []) or (key[:4] == 'time')):
                            to_del = False


                    if to_del:    
                        keys_to_del.append(key)

                for key in keys_to_del:
                    del self.__mdf__[key]


        if resample > 0.0:
            self.resample( resample )

        self._load(filterFunctionName)


    def _getChannels(self, channels):
        return [self[_channel] if isinstance(_channel, str) else _channel for _channel in list(flatten(channels))]

              
    #def plot(self, fnc_x=lambda x: x, fnc_y=lambda y:y, ax=None, f=None, ls='-', 
    #         drawstyle='steps', lib='matplotlib',**figargs):

    def _prepare_plot(self, channels, fnc_x = lambda x:x, fnc_y = lambda y:y,
             xrange=None, shift_time = False, label=False, drawstyle='steps',
             lib='matplotlib', legend_location=(0,0), 
             color_palette=None, **figargs):

        if not isinstance(channels, (list, np.ndarray, tuple)):
            channels = list(channels)

        num_channels = [len(_channel) 
                        if isinstance(_channel, (list, tuple, np.ndarray))
                        else 1 
                        for _channel in channels]
        num_axs = len(num_channels)

             
        if not all( (isinstance(_channel, str))or(isinstance(_channel, channel)) for _channel in list(flatten(channels))):
            raise ValueError("All channels arguments to plot must be string or channel subclasses.")
            return

        channels = self._getChannels(channels)
        eplt.set_plottinglib(lib)
        f, axs = eplt.make_figure(nrows=num_axs, ncols=1, axs=None, **figargs)
        i_channel = 0

        if color_palette is None:
            color_palette = bosch_standard_palette


        for i_row in range(num_axs):
            ls = []
            labels = []
            for _ in range(num_channels[i_row]):
                
                ls.append(channels[i_channel].plot(fnc_x, fnc_y, axs[i_row], f,
                                                    None, None, None, '-', color_palette[i_channel],
                                                    drawstyle, lib)[1])
                labels.append(channels[i_channel].longname)
                i_channel += 1

            eplt.dolegend(axs[i_row], ls, labels, legend_location)


        return axs


    def plot(self, channels, fnc_x = lambda x:x, fnc_y = lambda y:y,
             xrange=None, shift_time = False, label=False, drawstyle='steps',
             lib='matplotlib', output='notebook', legend_location=(0,0), 
             color_palette=None, **figargs):
        
        boCursor = figargs.pop('Cursor', False)
        boRangeSelect = figargs.pop('RangeSelect', False)
        axs = self._prepare_plot(channels, fnc_x, fnc_y ,xrange, shift_time, label, drawstyle,
                            lib, legend_location, color_palette, **figargs)

        return eplt.doshow(axs, widgets=None, ncols=1, output=output, Cursor=boCursor, 
                            RangeSelect=boRangeSelect)
            
        
        
        
    def calc(self, key, fnc, device='Calc', *func_args, **func_kwargs):
        _args = []
        for a in np.atleast_1d(args):
            if isinstance(a,(str, bytes)):
                _args.append(self[a])
            else:
                _args.append(a)

        dev, name = (device,key)
        chan = _args[0]
        description = kwargs.pop('description', r'calculated ...')
        unit = kwargs.pop('unit', r'a.u.')
        cname = dev + '/' + name
        master = chan.master

        _mdf = {cname:dict(data=fnc(*_args, **kwargs),
                                description = chan.description,
                                deviceName=dev, 
                                master=chan.master, 
                                masterType=chan.masterType, 
                                unit=chan.unit,
                                conversion=chan.conversion),
                master:dict(data=chan.x,
                            description = '',
                            deviceName=chan.master, 
                            master=chan.master, 
                            masterType=chan.masterType, 
                            unit='s')}
        #cname, mdf, name, longname, device
        self[cname] = calcchannel(cname=cname, 
                                  mdf=_mdf,
                                  name=name,
                                  longname=name+'('+dev+')',
                                  device=dev)
        self.names[name+ '_' + dev + '_'] = cname

        
    
    def findKeys(self, filter = '.*'):
        import re
        ret = []
        filter = np.atleast_1d(filter)
        for dk in self.keys():
            for ck in self[dk].keys():
                for kf in filter:
                    if re.match(kf, ck) is not None:
                        ret.append(dk+'/'+ck)
                        break
                
        return ret
    
        
        
    def _filterChannels( self, key, filter = '.*' ):
        import re
        for kf in np.atleast_1d(filter):
            if re.findall(kf, key) != []:
                return True
            
        return False


    def resample(self, samplingTime=None, masterChannel=None, kind='nearest'):
        """ Resamples all data groups into one data group having defined
        sampling interval or sharing same master channel

        Parameters
        ----------------
        samplingTime : float, optional
            resampling interval, None by default. If None, will merge all datagroups
            into a unique datagroup having the highest sampling rate from all datagroups
        **or**
        masterChannel : str, optional
            master channel name to be used for all channels

        Notes
        --------
        1. resampling is relatively safe for mdf3 as it contains only time series.
        However, mdf4 can contain also distance, angle, etc. It might make no sense
        to apply one resampling to several data groups that do not share same kind
        of master channel (like time resampling to distance or angle data groups)
        If several kind of data groups are used, you should better use pandas to resample

        2. resampling will convert all your channels so be careful for big files
        and memory consumption
        """
        try:
            self.__mdf__.resample(samplingTime, masterChannel, kind)
        except:
            pass#for dev in self.values():


    def exportToCSV(self, filename=None, sampling=None):
        """Exports mdf data into CSV file

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        sampling : float, optional
            sampling interval. None by default

        Notes
        --------
        Data saved in CSV fille be automatically resampled as it is difficult to save in this format
        data not sharing same master channel
        Warning: this can be slow for big data, CSV is text format after all
        """
        self.__mdf__.exportToCSV(filename, sampling)

    def exportToNetCDF(self, filename=None, sampling=None):
        """Exports mdf data into netcdf file

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        sampling : float, optional
            sampling interval.

        Dependency
        -----------------
        scipy
        """
        self.__mdf__.exportToNetCDF(filename, sampling)

    def exportToHDF5(self, filename=None, sampling=None):
        """Exports mdf class data structure into hdf5 file

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        sampling : float, optional
            sampling interval.

        Dependency
        ------------------
        h5py

        Notes
        --------
        The maximum attributes will be stored
        Data structure will be similar has it is in masterChannelList attribute
        """
        self.__mdf__.exportToHDF5(filename, sampling)
        
    def exportToMatlab(self, filename=None):
        """Export mdf data into Matlab file format 5, tentatively compressed

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        Dependency
        ------------------
        scipy

        Notes
        --------
        This method will dump all data into Matlab file but you will loose below information:
        - unit and descriptions of channel
        - data structure, what is corresponding master channel to a channel. Channels might have then different lengths
        """
        # export class data struture into .mat file
        self.__mdf__.exportToMatlab(filename)

    def exportToExcel(self, filename=None):
        """Exports mdf data into excel 95 to 2003 file

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        Dependencies
        --------------------
        xlwt for python 2.6+
        xlwt3 for python 3.2+

        Notes
        --------
        xlwt is not fast even for small files, consider other binary formats like HDF5 or Matlab
        If there are more than 256 channels, data will be saved over different worksheets
        Also Excel 2003 is becoming rare these days, prefer using exportToXlsx
        """
        self.__mdf__.exportToExcel(filename)

    def exportToXlsx(self, filename=None):
        """Exports mdf data into excel 2007 and 2010 file

        Parameters
        ----------------
        filename : str, optional
            file name. If no name defined, it will use original mdf name and path

        Dependency
        -----------------
        openpyxl

        Notes
        --------
        It is recommended to export resampled data for performances
        """
        self.__mdf__.exportToXlsx(filename)

    def convertToPandas(self, sampling=None):
        """converts mdf data structure into pandas dataframe(s)

        Parameters
        ----------------
        sampling : float, optional
            resampling interval

        Notes
        --------
        One pandas dataframe is converted per data group
        Not adapted yet for mdf4 as it considers only time master channels
        """
        # convert data structure into pandas module
        self.__mdf__.convertToPandas(sampling)

    def exportToPandas(self, sampling=None):
        """converts mdf data structure into pandas dataframe(s)

        Parameters
        ----------------
        sampling : float, optional
            resampling interval

        Notes
        --------
        One pandas dataframe is converted per data group
        Not adapted yet for mdf4 as it considers only time master channels
        """
        # convert data structure into pandas module

        
        from numpy import nan, datetime64, array
        try:
            import pandas as pd
        except:
            raise ImportError('Module pandas missing')
        _mdf = dotdict()
        masterGroups = []
        if sampling is not None:
            self.__mdf__.resample(sampling)
        if self.__mdf__.file_metadata['date'] != '' and self.__mdf__.file_metadata['time']!='':
            date = self.__mdf__.file_metadata['date'].replace(':', '-')
            time = self.__mdf__.file_metadata['time']
            datetimeInfo = datetime64(date + 'T' + time)
        else:
            datetimeInfo = datetime64(datetime.now())
        originalKeys = list(self.__mdf__.keys())
        # load DATA before converting to pandas
        if self.__mdf__._noDataLoading:
            if self.__mdf__.MDFVersionNumber < 400:  # up to version 3.x not compatible with version 4.x
                self.__mdf__._getAllChannelData3()
            else:  # MDF version 4.x
                self.__mdf__._getAllChannelData4()
        for group in list(self.__mdf__.masterChannelList.keys()):
            if group not in self.__mdf__.masterChannelList[group]:
                print('no master channel in group ' + group, file=stderr)
            elif self.__mdf__.getChannelMasterType(group)!=1:
                print('Warning: master channel is not time, not appropriate conversion for pandas', file=stderr)
            temp = {}
            # convert time channel into timedelta
            if group in self.__mdf__.masterChannelList[group]:
                try:
                    time = datetimeInfo + array(self.__mdf__.getChannelData(group) * 1E6, dtype='timedelta64[us]')
                    for channel in self.__mdf__.masterChannelList[group]:
                        data = self.__mdf__.getChannelData(channel)
                        if data.ndim == 1 and data.shape == time.shape:
                            temp[channel] = pd.Series(data, index=time)
                    _mdf[group + '_group'] = pd.DataFrame(temp)
                    _mdf[group + '_group'].pop(group)  # delete time channel, no need anymore
                except:
                    print('Warning: Converting Group', str(group), 'failed', file=stderr)
            else: # no master channel in channel group
                for channel in self.__mdf__.masterChannelList[group]:
                    data = self.__mdf__.getChannelData(channel)
                    masterGroups[channel] = group + '_group'
                    if data.ndim == 1:
                        temp[channel] = pd.Series(data)
                _mdf[group + '_group'] = pd.DataFrame(temp)
        # clean rest of self from data and time channel information
        
        # save time groups name in list
        #[masterGroups.append(group + '_group') for group in list(self.__mdf__.masterChannelList.keys())]
        return _mdf, masterGroups

    
def getEvents( data, filter, cfilter=['.*',]):
        '''
        Two filter-conditions are needed. One for the start-index of the slice
        and one for the end-index.
        Ex.:
        ----
            x_x_x_x_x_y_y_y_y_y_y_y_y_y_y_x_x_x_x_x_x_x_x
                     ^                   ^
                     |                   |
                  start-idx          end-idx
                  
        '''
        from ETKEvents import ETKEvents
        keys = data.findKeys( cfilter )
        _all = ETKEvents(data)
        
        # START
        try:
            ks, filt_func, lshift = filter[0]
        except:
            ks, filt_func = filter[0]
            lshift = 0

        ks = np.atleast_1d(ks)
        new_ks = []
        for k in ks:
            if not r'/' in k:
                k = data._getDeviceChannelName(k)
            new_ks.append(k)
        ks = np.array(new_ks)
        
        
        filt_func = np.atleast_1d(filt_func)
        
        idx = [f( np.array( data[k1].y, dtype=float) ) for f,k1 in zip(filt_func,ks)]
        try:
            idxs_s = np.where(np.all(idx, axis=0))[0]
        except:
            idxs_s = np.array( [] )
        
        # END
        try:
            ke, filt_func, rshift = filter[1]
        except:
            ke, filt_func = filter[1]
            rshift = 0
        ke = np.atleast_1d(ke)
        filt_func = np.atleast_1d(filt_func)
        
        idx = [f( np.array( data[k1].y, dtype=float) ) for f,k1 in zip(filt_func,ke)]
        try:
            idxs_e = np.where(np.all(idx, axis=0))[0]
        except:
            idxs_e = np.array([])
            
        filt_time_s = data[ks[0]].master
        filt_time_e = data[ke[0]].master
        ref_time_s = data[ks[0]].x
        ref_time_e = data[ke[0]].x
        
        # cut idxs
        if filt_time_s == filt_time_e:
            try:
                i_dx1 = np.where(idxs_s < idxs_s[0])[0][-1]
            except:
                i_dx1 = 0
            try:
                i_dx2 = np.where(idxs_s < idxs_e[-1])[0][-1]
            except:
                i_dx2 = 10000
            idxs_s = idxs_s[i_dx1:i_dx2+1]
            
            try:
                i_dx1 = np.where(idxs_e > idxs_s[0])[0][0]
            except:
                i_dx1 = 10000
            try:
                i_dx2 = np.where(idxs_e > idxs_s[-1])[0][0]
            except:
                i_dx2 = len(idxs_s)
            idxs_e = idxs_e[i_dx1:i_dx2+1]
            
            
            #while 1:
            #    try:
            #        i_dx = np.where( idxs_s >= idxs_e)[0][0]
            #        idxs_e = np.delete( idxs_e, i_dx )
            #    except:
            #        break
            tmp = []
            for ix in range( len( idxs_s ) ):
                try:
                    i_d = np.where( idxs_e > idxs_s[ix])[0][0]
                    tmp.append( idxs_e[i_d] )
                except:
                    break

            idxs_s = np.delete( idxs_s, np.where( np.diff( tmp ) == 0 )[0] ) + lshift
            idxs_e = np.delete( tmp, np.where( np.diff( tmp ) == 0 )[0] ) + rshift
            
        print( '#Measurments: ', len( idxs_s ) )
        masterTimes_s = {filt_time_s:idxs_s}
        masterTimes_e = {filt_time_e:idxs_e}

        if len( idxs_s ) > 0:
            for k in keys:
                if k.find('time') < 0:
                    if data[k].master not in masterTimes_s:
                        masterTimes_s[data[k].master] = [ np.where( data[k].x >= ref_time_s[idx])[0][0] for idx in idxs_s ]
                    if data[k].master not in masterTimes_e:
                        masterTimes_e[data[k].master] = [ np.where( data[k].x <= ref_time_e[idx])[0][-1] for idx in idxs_e ]
                        
                    _all.add(data[k], [list(masterTimes_s[data[k].master]), list(masterTimes_e[data[k].master])])
                    
        return _all

def merge(data1, data2):
    for dev1 in data1.values():
        for chan1 in dev1.values():
            chan2 = data2[chan1.device][chan1.name]
            data1.__mdf__[chan1._cname]['data'] = np.concatenate((chan1['data'],chan2['data']))
            if chan1['time'].shape !=  chan1['data'].shape:
                tmp = chan2['time'] + chan1['time'][-1]
                data1.__mdf__[chan1.master]['data'] = np.concatenate((chan1['time'],tmp))
