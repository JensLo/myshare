# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from funcs import *
from math import ceil
from matplotlib.axes import Subplot
try:
    from bokeh import plotting as boplt
    from bokeh.layouts import column, row, widgetbox, Spacer
    from bokeh.models import (ColumnDataSource, Legend, ToolbarBox, PanTool, WheelZoomTool, 
                              SaveTool, ResetTool, BoxZoomTool, UndoTool, RedoTool)
    from bokeh.io import push_notebook
    _BOKELOADED_ = True

    _BOKETOOLS_ = 'xpan, xwheel_zoom, xbox_zoom, undo, redo, reset'

except:
    _BOKELOADED_ = False

from rcParams_bosch import *

_MATPLOTLIB_ = ('matplotlib', 'pylab', 'pyplot')
_PLOTTINGLIB_ = _MATPLOTLIB_[0]

def slice_equal(x,y):
    x = x[:len(y)]
    y = y[:len(x)]
    return (x,y)

def _process_steps(x,y):
    #xs = np.empty(2*len(x)-1)
    #xs[::2] = x[:]
    #xs[1::2] = x[1:]

    #ys = np.empty(2*len(y)-1)
    #ys[::2] = y[:]
    #ys[1::2] = y[:-1]
    x,y = x[::2], y[::2]
    xs = np.empty(2*len(x)-1)
    xs[::2] = x[:]
    xs[1::2] = x[1:]

    ys = np.empty(2*len(y)-1)
    ys[::2] = y[:]
    ys[1::2] = y[:-1]

    return xs, ys

def set_plottinglib(lib):
    global _PLOTTINGLIB_, _MATPLOTLIB_
    if lib.lower() in _MATPLOTLIB_:
        _PLOTTINGLIB_ = _MATPLOTLIB_[0]
    else:
        _PLOTTINGLIB_ = 'bokeh'


def _make_pyfigure(nrows=1, ncols=1, axs=None, **figargs):
    if axs is None:
        f, axs = plt.subplots(nrows, ncols, squeeze=False, sharex=True, **figargs)
        return f, list(flatten(axs))
    else:
        if not isinstance(axs, Iterable): axs = [axs]
        axs = list(flatten(axs))
        f = axs[0].get_figure()

        return f, axs

def _make_bofigure(nrows=1, ncols=1, axs=None, **figargs):
    if not _BOKELOADED_:
        raise Warning('Bokeh library is not installed')
        return

    if axs is None:  
        nplts = nrows  * ncols
        axs = []
        for i in range(nplts):
            f = boplt.figure(tools=_BOKETOOLS_, toolbar_location=None, logo=None,  min_border_bottom=0, **figargs)
            axs.append(f)

        return None, axs

    else:
        if not isinstance(axs, Iterable): axs = [axs]
        axs = list(flatten(axs))
        return None, axs

def _show_bofigure(axs, widgets, ncols=1, output='notebook', **kwargs):
    if output.lower() == 'notebook':
        boplt.output_notebook()
        notebook_handle=True
    elif output.lower() == 'file':
        import tempfile
        boplt.output_file(tempfile.mktemp('.html'))
        notebook_handle=False
    else:
        doc = boplt.curdoc()
        notebook_handle=False

    raw_tools = []
    for ax in axs:
        raw_tools = raw_tools + ax.toolbar.tools
        ax.x_range = axs[0].x_range

    for ax in axs[:-1]:
        ax.xaxis.visible=False

    axs[-1].plot_height += 15

    if kwargs.pop('RangeSelect', False):
        rs = make_RangeSelect(axs)
    else:
        rs = Spacer(width=1, height=1)

    if kwargs.pop('Cursor', False):
        button = make_cursors(axs, )
    else:
        button = None


    toolbar = ToolbarBox(tools=raw_tools,
                        logo=None, toolbar_location='right', merge_tools=True, 
                        sizing_mode = 'stretch_both')

    sizing_mode=None
    if button is None:
        if rs is None:
            col = column(*axs)#, sizing_mode='scale_width')
        else:
            col = column(*axs, rs)#, sizing_mode='scale_width')
    else:
        if rs is None:
            col = column(button, *axs)#, sizing_mode='scale_width')
        else:
            col = column(button, *axs, rs)#, sizing_mode='scale_width')

    #layout = row(col, Spacer(width=40, height=300),toolbar, sizing_mode='stretch_both')
    layout = row(col, Spacer(width=40, height=300),toolbar)

    if output.lower() in ('notebook', 'file'):
        return boplt.show(layout,  notebook_handle=notebook_handle), layout, axs
    else:
        doc.add_root(layout)
        return None

def _pyplot(x, y, ax, label=None, xlabel=None, ylabel=None, 
            ls='-', drawstyle='steps', color=None, **args):
  
    l = ax.plot(*(slice_equal(x, y)), ls=ls, drawstyle=drawstyle, label=label, c=color, **args)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend(loc=1, frameon = True, fancybox=True, 
                   framealpha=0.6, facecolor='#ffffff', 
                   edgecolor='#aaaaaa')
    return l

def _boplot(x, y, ax, label=None, xlabel=None, ylabel=None, 
            ls='-', drawstyle='steps', color=None, **args):

    
    if not _BOKELOADED_:
        raise Warning('Bokeh library is not installed')
        return

    if drawstyle == 'steps':
        x,y = _process_steps(x,y)

    if len(x>10*1500):
        ratio = ceil(len(x) / (10*1500))
    else:
        ratio = 1

    ratio = 1

    _x = x[::ratio]
    _y = y[::ratio]
    source = ColumnDataSource(data=dict(x=_x, y=_y))

    label=None
    if label is not None:
        l = ax.line('x', 'y', legend=label, line_width=3, line_color=color, 
                     source=source, **args)
        ax.legend.click_policy = 'hide'
    else:
        l = ax.line('x', 'y', line_width=3, line_color=color, 
                     source=source, **args)
    if xlabel is not None:
         ax.xaxis.axis_label = xlabel
    if ylabel is not None:
        ax.yaxis.axis_label = ylabel

    return l


def _pylegend(ax, ls, labels, location=1):
    ax.legend(loc=location, frameon = True, fancybox=True, 
                   framealpha=0.6, facecolor='#ffffff', 
                   edgecolor='#aaaaaa')

def _bolegend(ax, ls, labels, location=(0, 0)):
    legend = Legend(items=[(ll,[l]) for ll, l in zip(labels, ls)], location=location)
    legend.click_policy = 'hide'
    ax.add_layout(legend, 'right')


def make_figure(nrows=1, ncols=1, axs=None, **figargs):
    if _PLOTTINGLIB_.lower() in _MATPLOTLIB_:
        return _make_pyfigure(nrows, ncols, axs, **figargs)
    else:
        return _make_bofigure(nrows, ncols, axs, **figargs)


def doplot(x, y, ax, label=None, xlabel=None, ylabel=None, 
           ls='-', drawstyle='steps', color=None, **args):
    if _PLOTTINGLIB_.lower() in _MATPLOTLIB_:
        args.pop('name')
        return _pyplot(x, y, ax, label, xlabel, ylabel, ls, 
                       drawstyle, color, **args)
    else:
        return _boplot(x, y, ax, label, xlabel, ylabel, ls, 
                       drawstyle, color, **args)

def doshow(axs=None, widgets=None, ncols=1, output='notebook', **kwargs):
    if _PLOTTINGLIB_.lower() in _MATPLOTLIB_:
        return plt.show()
    else:
        if axs is None:
            raise Warning('Bokeh model for argument ax is required!')
        return _show_bofigure(axs, widgets, ncols, output, **kwargs)

def dolegend(ax, ls, labels, location=None):
    if _PLOTTINGLIB_.lower() in _MATPLOTLIB_:
        if location is None:
            location=1
        _pylegend(ax, ls, labels, location)
    else:
        if location is None:
            location=(0,-10)

        _bolegend(ax, ls, labels, location)




########### BOKEH EXTENSION ############################

from bokeh.plotting import figure, show, output_notebook, Figure
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Button, Span, CustomJS, ColumnDataSource, Label, LabelSet,BoxSelectTool, BoxAnnotation
from bokeh import events
from bokeh.core.properties import Int, Bool

class Cursor(Span):
    active = Bool
    __subtype__ = 'Cursor'
    __view_model__ = 'Span'
    def __init__(self, **kwargs):
        
        super(Cursor, self).__init__(**kwargs)

class RangeSelect(BoxAnnotation):
    active_mode = Int
    active = Bool
    __subtype__ = 'RangeSelect'
    __view_model__ = 'BoxAnnotation'
    def __init__(self, **kwargs):
        
        super(RangeSelect, self).__init__(**kwargs)



## CUSTOM CALLBACKS
cursor_label_callback_code = """
        function setLabel(_x, _y, _data, _idx,
                         _xoff, _yoff, _color, _label) {
            _data['xs'].push(_x);
            _data['ys'].push(_y);
            var t = ''
            if (_idx < 0){
                t = 'x = '+Number(_x).toFixed(4);
            }else{
                t = _label+ ' = ' +Number(_y).toFixed(3); 
            }
            if(_xoff == -1) _xoff = -t.length*6.1;
            _data['texts'].push(t);
            _data['xoff'].push(_xoff);
            _data['yoff'].push(_yoff);
            _data['colors'].push(_color)
            //console.log(t, _xoff, _yoff, _x,_y)
            
        }
        // get data source from Callback args
        var ladata = label_source.data;
      
        /// reset source
        ladata['xs'] = [];
        ladata['ys'] = [];
        ladata['texts'] = [];
        ladata['xoff'] = [];
        ladata['yoff'] = [];
        ladata['colors'] = [];
        
        /// update data source with new Rect attributes
        
        // CURSOR 1
        // add x-labels
        var i = -1;
        var x = 0;
        var y = 0;
        var xoff = 1;
        if( (c1.location < c2.location)) xoff = -1;
        //if( (c1.name == 'CURSOR1')&&(c1.location < c2.location)) xoff = -1;
        //if( (c1.name == 'CURSOR2')&&(c1.location < c2.location)) xoff = -1;
        x = c1.location;
        y = fig.y_range.start;
        //!!setLabel(x, y, ladata, -1, xoff, -20, '#525F6B', 'x'); //<-- //!! is removed for lowest figure 
        // add y-labels
        for(key in fig.renderers){
            if(fig.renderers[key].glyph){
                var data = fig.renderers[key].data_source.data;
                var xkey = 'x'; //Object.keys(data)[0];
                var ykey = 'y'; //Object.keys(data)[1];
                var color = fig.renderers[key].glyph.line_color.value;
                var label = fig.renderers[key].name;
                function get_idx(_x) {
                    return _x >= x;
                };
                _idx = data[xkey].findIndex(get_idx);
                y = data[ykey][_idx];
                i = i+1;
                setLabel(x, y, ladata, i, xoff, 0, color, label);  
            }
        }
        
        // CURSOR 2
        // add x-labels
        var i = -1;
        var x = 0;
        var y = 0;
        xoff = -1*xoff;
        x = c2.location;
        y = fig.y_range.start;
        //!!setLabel(x, 0, ladata, -1, xoff, -20, '#525F6B'); //<-- //!! is removed for lowest figure 
        // add y-labels
        for(key in fig.renderers){
            if(fig.renderers[key].glyph){
                var data = fig.renderers[key].data_source.data;
                var xkey = 'x'; Object.keys(data)[0];
                var ykey = 'y'; Object.keys(data)[1];
                var color = fig.renderers[key].glyph.line_color.value;
                var label = fig.renderers[key].name;
                function get_idx(_x) {
                    return _x >= x;
                };
                _idx = data[xkey].findIndex(get_idx);
                y = data[ykey][_idx];
                i = i+1;
                setLabel(x, y, ladata, i, xoff, 0, color, label);  
            }
        }
        // add dx-label
        //!!var dx = Math.abs(c1.location - c2.location); //<-- //!! is removed for lowest figure 
        //!!dx_label.text = 'dx = ' + Number(dx).toFixed(4);
        //!!if (c1.location < c2.location) {
        //!!    dx_label.x = c1.location + dx/2 ;
        //!!}
        //!!else{
        //!!    dx_label.x = c1.location - dx/2 ;
        //!!}
        //!!dx_label.x_offset = -dx_label.text.length*3.;
        //!!dx_label.y = fig.y_range.start;
        //!!dx_label.y_offset = 0;
        // emit update of data source
        label_source.change.emit();
        //!!dx_label.change.emit();
    """

show_cursor_code="""
    
    var c1s = [ %s ];
    var c2s = [ %s ];
    var c1labels = [ %s ];
    var c2labels = [ %s ];
    for (idx in c1s){
        var c1 = c1s[idx];
        var c2 = c2s[idx];
        var c1lab = c1labels[idx];
        var c2lab = c2labels[idx];
        if(c1.visible){
            c1.visible = false;
            c2.visible = false;
            c1.active = false;
            c2.active = false;
            c1lab.visible = false;
            c2lab.visible = false;
            dxlab.visible = false;
        }
        else{
            c1.visible = true;
            c2.visible = true;
            c1.active = false;
            c2.active = false;
            c1lab.visible = true;
            c2lab.visible = true;
            dxlab.visible = true;
            if( (c1.location == -1000) ||
                (c1.location < x_range.start) ||
                (c1.location > x_range.end) ){
                var dx = x_range.end - x_range.start;        
                c1.location = x_range.start + dx / 3;
            }
            if( (c2.location == -1000) ||
                (c2.location < x_range.start) ||
                (c2.location > x_range.end) ){
                var dx = x_range.end - x_range.start;        
                c2.location = x_range.start + dx / 1.5;
            }
        }
    }   
    """

activate_cursor_code="""
    var x = cb_obj['x'];
    dx = xrange.end - xrange.start;
    var c1s = [ %s ];
    var c2s = [ %s ];
    for (idx in c1s){
        var c1 = c1s[idx];
        var c2 = c2s[idx];
        if (Math.abs(x-c1.location)<dx/200){
            c1.active = (c1.active == false) && (c2.active == false);
        }    
        if (Math.abs(x-c2.location)<dx/200){
            c2.active = (c2.active == false) && (c1.active == false);
        }
    }
    """

move_cursor_code="""
    var c1s = [ %s ];
    var c2s = [ %s ];
    for (idx in c1s){
        var c1 = c1s[idx];
        var c2 = c2s[idx];
        var x = cb_obj['x'];
        if (c1.active){
            c1.location = x;
        }
            
        if (c2.active){
            c2.location = x;
        }
    }
    """  

activate_rangeselect_code="""
        console.log('activate_rangeselect_code')
        var x = cb_obj.x;
        dx = rangeSelect.right - rangeSelect.left;
        
        if (rangeSelect.active_mode > -1) {
            rangeSelect.active_mode = -1;
        }
        else if ((x > rangeSelect.left) && (x < rangeSelect.right)){
        
            rangeSelect.active_mode = 0;
            if (x-rangeSelect.left<dx/5){
                rangeSelect.active_mode = 1;
            }
            if (rangeSelect.right-x<dx/5){
                rangeSelect.active_mode = 2;
            }
        }
    """

move_rangeselect_code="""
        console.log('move_rangeselect_code')
        var x = cb_obj.x;
        width = rangeSelect.right - rangeSelect.left;
        if (rangeSelect.active_mode == 0){
            rangeSelect.left = x - width/2;
            rangeSelect.right = x + width/2;
            p1.x_range.start = rangeSelect.left;
            p1.x_range.end = rangeSelect.right;
        }else if (rangeSelect.active_mode == 1){
            rangeSelect.left = x;
            p1.x_range.start = rangeSelect.left;
            p1.x_range.end = rangeSelect.right;
        }else if (rangeSelect.active_mode == 2){
            rangeSelect.right = x;
            p1.x_range.start = rangeSelect.left;
            p1.x_range.end = rangeSelect.right;
        }
        console.log(rangeSelect.left, rangeSelect.right, p1.x_range.start, p1.x_range.end, rangeSelect.active_mode)
    """  

adjust_range_select_code="""
    console.log('adjust_range_select_code')
    if(rangeSelect.active_mode === undefined) {
        rangeSelect.active_mode = -1;
    }
    if ( rangeSelect.active_mode == -1){
        rangeSelect.left = cb_obj.start;
        rangeSelect.right = cb_obj.end;
    }

"""

boxSelsct_code = """
        rangeSelect.left = cb_data.geometry.x0;
        rangeSelect.right = cb_data.geometry.x1;
        p1.x_range.start = cb_data.geometry.x0;
        p1.x_range.end = cb_data.geometry.x1;
        """


dataCallback_code = """

    var sources = [ %s ];
    var full_sources = [ %s ];
   
    for (idx in sources){
        var source = sources[idx];
        var full_source = full_sources[idx];
        var data = source['data'];
        var full_data = full_source['data'];
        var xrange = cb_obj;
        var time_start = xrange['start'];
        var time_end = xrange['end'];

        var time = full_data.x;
        var vals = full_data.y;
        t_idx_start = time.filter(function(st){return st>=time_start})[0];
        t_idx_start = time.indexOf(t_idx_start);

        t_idx_end = time.filter(function(st){return st>=time_end})[0];
        t_idx_end = time.indexOf(t_idx_end);
        if (t_idx_end == -1) t_idx_end = time.length;

        var fw = fig.width;

        var new_time = time.slice(t_idx_start, t_idx_end);
        if(new_time.length >  2*fw) {
            var ratio = Math.ceil(new_time.length / (fw));
            new_time = new_time.filter(function(value, index, Arr) {
                return index %% ratio == 0;
            });
        }
        new_time = new_time.filter(function(st){return !isNaN(st)});
        var new_vals = vals.slice(t_idx_start, t_idx_end);
        if(new_vals.length >  2*fw) {
            var ratio = Math.ceil(new_vals.length / (fw));
            new_vals = new_vals.filter(function(value, index, Arr) {
                return index %% ratio == 0;
            });
        }
        new_vals = new_vals.filter(function(st){return !isNaN(st)});

        data['x'] = new_time;
        data['y'] = new_vals;

        source.change.emit();
    }

"""


def make_cursors(axs, xs=[None, None], button=None): 
    global cursor_label_callback_code 
    cursors = []
    labels = []
    
    if not isinstance(axs, (list, np.ndarray, tuple)):
        axs = [axs]

    if not all(isinstance(fig, Figure) for fig in axs):
        raise ValueError("All figure arguments to make_cursors must be Figure subclasses.")
        return

    for fig in axs:
        _x0 = xs[0] if xs[0] is not None else -1000
        c1 = Cursor(location=_x0, line_width=2, 
                 line_color=bosch_gray_palette[0], dimension='height', 
                 name='CURSOR1', active=False, visible=False)
        _x1 = xs[1] if xs[1] is not None else -1000
        c2 = Cursor(location=_x1, line_width=2, 
                 line_color=bosch_gray_palette[0], dimension='height', 
                 name='CURSOR2', active=False, visible=False)

        
        label1_source = ColumnDataSource(data=dict(xs=[],ys=[],texts=[], 
                                                   xoff=[], yoff=[], colors=[]))
        label2_source = ColumnDataSource(data=dict(xs=[],ys=[],texts=[], 
                                                   xoff=[], yoff=[], colors=[]))

        c1labels = LabelSet(x='xs', y='ys', x_offset='xoff', y_offset='yoff', 
                            text='texts', level='overlay', source=label1_source, 
                            render_mode='canvas', text_font_size='9pt', text_color='white',
                            border_line_color=bosch_gray_palette[1], border_line_width=1,
                            background_fill_color = 'colors', 
                            background_fill_alpha=0.9  )
        c2labels = LabelSet(x='xs', y='ys', x_offset='xoff', y_offset='yoff', 
                            text='texts', level='overlay', source=label2_source, 
                            render_mode='canvas', text_font_size='9pt', text_color='white',
                            border_line_color=bosch_gray_palette[1], border_line_width=1,
                            background_fill_color = 'colors', 
                            background_fill_alpha=0.9  )
        dx_label = Label(x=-1000, y=0, x_offset=0, y_offset=-10, text='dx = ', level='overlay', 
                         render_mode='canvas', text_font_size='9pt', text_color='white',
                         border_line_color=bosch_gray_palette[1], border_line_width=1,
                         background_fill_color = bosch_gray_palette[0], 
                         background_fill_alpha=0.9  )
        

        cursors.append([c1, c2])
        labels.append([c1labels, c2labels])

        fig.add_layout(c1)
        fig.add_layout(c2)
        fig.add_layout(c1labels)
        fig.add_layout(c2labels)

        if fig == axs[-1]:
            cursor_label_callback_code = cursor_label_callback_code.replace('//!!','')
            fig.add_layout(dx_label)

        c1.js_on_change('location',CustomJS(args=dict(label_source=label1_source, 
                                                      fig=fig,
                                                      c1=c1, 
                                                      c2=c2, 
                                                      dx_label=dx_label), 
                                                 code=cursor_label_callback_code))
        c2.js_on_change('location',CustomJS(args=dict(label_source=label1_source, 
                                                      fig=fig,
                                                      c1=c2, 
                                                      c2=c1, 
                                                      dx_label=dx_label), 
                                                 code=cursor_label_callback_code))
    
        
        
    c1_dict = {'c1_%d'%i:c[0] for i, c in enumerate(cursors)}
    c2_dict = {'c2_%d'%i:c[1] for i,c in enumerate(cursors)}
    labs1_dict = {'l1_%d'%i:l[0] for i,l in enumerate(labels)}
    labs2_dict = {'l2_%d'%i:l[1] for i,l in enumerate(labels)}
    
    arg_dict = dict()
    arg_dict.update(c1_dict)
    arg_dict.update(c2_dict)
    arg_dict.update(labs1_dict)
    arg_dict.update(labs2_dict)
    arg_dict.update(dict(dxlab=dx_label,x_range=fig.x_range))
    if button is None:
             b = Button(label="Cursor aktivieren / deaktivieren", button_type="primary", 
                        callback=CustomJS(args=arg_dict, 
                                          code=show_cursor_code%(','.join(c1_dict.keys()),
                                                                 ','.join(c2_dict.keys()),
                                                                 ','.join(labs1_dict.keys()),
                                                                 ','.join(labs2_dict.keys()))))

    arg_dict = dict()
    arg_dict.update(c1_dict)
    arg_dict.update(c2_dict)
    arg_dict.update(dict(xrange=fig.x_range))
    for fig in axs:
        fig.js_on_event(events.MouseMove, CustomJS(args=arg_dict, 
                                                   code=move_cursor_code%(','.join(c1_dict.keys()),
                                                                          ','.join(c2_dict.keys()))))

        fig.js_on_event(events.Tap, CustomJS(args=arg_dict, 
                                             code=activate_cursor_code%(','.join(c1_dict.keys()),
                                                                          ','.join(c2_dict.keys()))))

    return b


def make_RangeSelect(axs, output=None, color=bosch_gray_palette[0]):
    p1 = axs[0]
    plot = figure(plot_width=p1.plot_width, plot_height=int(p1.plot_height/2), min_border_top=0, 
                 min_border_bottom=0, toolbar_location=None, tools='',logo=None)

    data = [ren.data_source for ren in p1.renderers if 'glyph' in ren.__properties__][0]

    over_view_data = ColumnDataSource(data=dict(x=data.data['x'][::10], y=data.data['y'][::10]))
    
    plot.line('x', 'y', source=over_view_data, line_color=bosch_gray_palette[1], line_width=2)
    plot.yaxis.visible = False
    plot.ygrid.grid_line_color = None
    plot.xaxis.axis_label = 'time / s'


    rs = RangeSelect(left = 0, right=100, fill_alpha=0.3,
                     line_color=color, fill_color=color)

    plot.add_layout(rs)

    plot.add_tools(BoxSelectTool(callback=CustomJS(args=dict(p1=p1, rangeSelect=rs), code=boxSelsct_code), dimensions='width', ))
    rs.js_on_event(events.Pan)
    

    #print(adjust_range_select_code + dataCallback_code%(','.join(sources_dict.keys())))

    if output is None:

        all_data = [ren.data_source for fig in axs for ren in fig.renderers if 'glyph' in ren.__properties__]
        new_data =[ColumnDataSource(data=dict(x=data.data['x'][:] ,y=data.data['y'][:])) for data in all_data]
        sources_dict = {'sources_%d'%i:d for i, d in enumerate(all_data)}
        data_dict = {'data_%d'%i:d for i, d in enumerate(new_data)}
        arg_dict = dict()
        arg_dict.update(sources_dict)
        arg_dict.update(data_dict)
        arg_dict.update(dict(rangeSelect=rs, fig=p1))

        p1.x_range.callback = CustomJS(args=arg_dict, 
                                       code=adjust_range_select_code + dataCallback_code%(','.join(sources_dict.keys()),
                                                                                          ','.join(data_dict.keys())))

    plot.js_on_event(events.Tap, CustomJS(args=dict(rangeSelect=rs, p1=p1 ),code=activate_rangeselect_code))
    plot.js_on_event(events.MouseMove, CustomJS(args=dict(rangeSelect=rs, p1=p1), code=move_rangeselect_code))
    rs.js_on_event(events.Pan)

       

    return plot, rs
