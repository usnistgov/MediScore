#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:26:11 2017

@author: tnk12
"""
import numpy as np
from functools import reduce
from itertools import cycle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class IntervalCompute():
    
    @staticmethod
    def gen_random_intervals(n, max_range, random_seed=None):
        """Generate a random array of intervals between 0 and max_range
        :param n: number of intervals
        :param max_range: maximum value of all intervals boundaries
        :Example:
        >>> gen_random_intervals(2, 100, random_seed=None)
        array([[30, 33], [47, 75]], dtype=uint32)
        """
        if random_seed: np.random.seed(random_seed)
        f_positions = np.sort(np.random.randint(max_range,  size=2*n, dtype=np.uint32)).reshape((n,2))
        return f_positions

    @staticmethod
    def timestamps_to_intervals(timestamps,dtype=np.uint64):
        """Function similar to itertools.pairwise but for numpy.array
        np.array([x0, x1, x2, x3]) -> np.array([[x0, x1], [x1, x2], [x2, x3]])
        :param dtype: specify the data type of the returned array
        :Example:
        >>> timestamps_to_intervals(np.array([1,3,5,7,9]),dtype=np.uint32) 
        array([[1, 3], [3, 5], [5, 7], [7, 9]], dtype=uint32)
        """
        ts_intervals = np.zeros(2 * (timestamps.size - 1), dtype=dtype)
        ts_intervals[1:-1] = np.repeat(timestamps[1:-1], 2)
        ts_intervals[0] = timestamps[0]
        ts_intervals[-1] = timestamps[-1]
        ts_intervals = ts_intervals.reshape(timestamps.size-1,2)
        return ts_intervals

    @staticmethod
    def truncate(interval_array, FrameCount):
        """Function that cut a list of interval at a specific value v
        :param interval_array: np.array([[x0, x1], ...])
        :FrameCount: value to truncate
        :returns: a truncated copy of the interval_array
        :Example:
        >>> truncate(np.array([[10,20],[30,40]]),25)
        array([[10, 20]])
        >>> truncate(np.array([[10,20],[30,40]]),30)
        array([[10, 20]])
        >>> truncate(np.array([[10,20],[30,40]]),35)
        array([[10, 20], [30, 35]])
        >>> truncate(np.array([[10,20],[30,40]]),40)
        array([[10, 20], [30, 40]])
        >>> truncate(np.array([[10,20],[30,40]]),50)
        array([[10, 20], [30, 40]])
        """
        i = interval_array.copy()
        mask_oversize = np.transpose(np.nonzero(interval_array >= FrameCount))
        if mask_oversize.size:
            # If there is an overlap
            if mask_oversize[0,1] == 1:
                i[mask_oversize[0,0],mask_oversize[0,1]] = FrameCount
                i = i[:mask_oversize[0,0]+1]
            else:
                i = i[:mask_oversize[0,0]]
        return i
        
    @staticmethod
    def compute_intervals_union(Intervals_list):
        """Compute the union of a list of intervals list
        :param Intervals_list: format -> [ np.array([[x0,x1],[x2,x3],...]), ... ]
        """
        assert isinstance(Intervals_list, list), "Parameter needs to be a list"
        if Intervals_list:
            # Filtering empty intervals first
            Intervals_list_filtered = [x for x in Intervals_list if x.size != 0]
            if Intervals_list_filtered:
                intervals_stack = np.sort(np.vstack(Intervals_list_filtered)) # We sort to avoid any issues in case x_i > x_i+1
                intervals = intervals_stack[intervals_stack[:,0].argsort()]
                union = [intervals[0].copy()]
                current_union = union[-1]
                for interval in intervals[1:]:
                    start, end = interval
                    # If there is overlap
                    if start <= current_union[1]:
                        # if the interval end after the current union, we extend the union
                        if end > current_union[1]:
                            current_union[1] = end
                    else:
                        union.append(interval.copy())
                        current_union = union[-1]
                return np.array(union)
            else:
                return np.array([[]])
        else:
            return np.array([[]])

    @staticmethod
    def get_complementary_union(intervals, global_interval, compute_mask_only=True):
        """Return the union of [l] and the complementary of l in global_interval
        (See examples)
        :param intervals: List of list intervals [[x0,x1], [x1,x2], ...]
        :param global_interval: interval list [start, end]
        :Example:
        >>> get_complementary_union([[5,15]], [-10,20], compute_mask_only=False)
        ([[-10, 5], [5, 15], [15, 20]], [0, 1, 0])
        >>> get_complementary_union(np.array([[5,15]]).tolist(), [-10,20], compute_mask_only=False)
        ([[-10, 5], [5, 15], [15, 20]], [0, 1, 0])
        >>> get_complementary_union([[5,15], [10,20]], [-10,30], compute_mask_only=False)
        ([[-10, 5], [5, 15], [10, 20], [20, 30]], [0, 1, 1, 0])
        >>> get_complementary_union(np.array([[5,15], [10,20]]), [-10,30]) # We return only the mask
        array([0, 1, 1, 0], dtype=uint8)
        """
        if isinstance(intervals, list): 
            assert (len(intervals) != 0), "intervals list is empty"  
            if len(intervals[0]) == 0: #Empty interval
                if compute_mask_only:
                    return np.array([0], dtype=np.uint8)
                else:
                    return ([global_interval], [0])
        else: 
            assert (intervals.shape != (0,)) , "intervals array is empty"  
            if intervals.shape == (1,0): #Empty interval
                if compute_mask_only:
                    return np.array([0], dtype=np.uint8)
                else:
                    return (np.array(global_interval), [0])

        start, boundary = global_interval
        # Efficient computation in the case we only compute the boolean mask
        if compute_mask_only:
            counter = 0
            if isinstance(intervals, list):
                mask = [0] * (2*len(intervals) + 1)
            else:
                assert (len(intervals.shape) == 2), "invalid intervals list dimensions, must be 2 dimensionnal"
                mask = np.zeros(2*intervals.shape[0] + 1, dtype=np.uint8)
            for interval in intervals:
                i_start, i_end = interval
                if start < i_start:
                    mask[counter] = 0
                    mask[counter+1] = 1
                    counter += 2
                else:
                    mask[counter] = 1
                    counter += 1
                start = i_end
            if i_end < boundary:
                mask[counter] = 0
                counter += 1
            return mask[:counter]
        else:
            L, mask = [], []
            if isinstance(intervals, list):    
                for interval in intervals:
                    i_start, i_end = interval
                    if start < i_start:
                        L.extend([[start,i_start],interval])
                        mask.extend([0,1])
                    else:
                        L.append(interval)
                        mask.append(1)
                    start = i_end
                if i_end < boundary:
                    L.append([i_end, boundary])
                    mask.append(0)
                return L, mask
            else:
                print("Error: Intervals type need to be a list.")

    @staticmethod
    def compute_collars(intervals, collar, crop_to_range=None):
        """Compute collar around interval boundaries
        :param intervals: intervals list to apply collars
        :param collars: a number 'col' or an np.array([[col0,col1],[col2,col3]) of shape (2,2)
        :param crop_to_range: if provided, collars intervals will be cropped to the range 
        :returns:
            if collar = col :
                [[x0, y0]] -> [[x0-col, x0+col],[y0-col, y0+col]]
            if collar = np.array([[col0,col1],[col2,col3])
                [[x0, y0]] -> [[x0-col0, x0+col1],[y0-col2, y0+col3]]
        :Example:
            >>> compute_collars(np.array([[10,20], [30,40]]), 2)
            array([[8, 12], [18, 22], [28, 32], [38, 42]])
            >>> compute_collars(np.array([[10,20]]), np.array([[1,2], [3,4]]))
            array([[9, 12], [17, 24]])
        """
        
        if isinstance(collar, int) or isinstance(collar, float):
            assert (collar > 0), "collar has to be strictly greater than 0"
            collars = np.tile([-collar,collar],(intervals.size,1)) + np.sort(intervals.ravel())[:,np.newaxis]
        elif len(collar) == 2:
            assert (np.sum(collar == 0) < 4), "at least one collar value has to be strictly greater than 0"
            assert (~np.any(collar < 0)), 'collar value must be strictly greater than 0'
            x = np.tile((collar * np.array([-1,1])).ravel(),(intervals.shape[0],1))
            y = np.repeat(intervals,[2,2], axis=1)
            collars = (x + y).reshape(intervals.size,2)
        else:
            print("Error in compute_collars: collar datatype not supported.")
            return None
            
        if crop_to_range is not None:
            start, end = crop_to_range
            collars[collars < start] = start
            collars[collars > end] = end
        return collars
    
    @staticmethod
    def aggregate_intervals(intervals_sequence, global_range, print_results=False):
        """Aggregation of a list of intervals lists. Returns the overlap code vector.
        Note: An intervals_list has this format : np.array([[x0,y0],[x1,y1],[x2,y2],...,[xn,yn]])
            * For every i, x_i != y_i 
        :param intervals_sequence: list of intervals lists ~ [np.array([[x0,x1], [x2,x3]]), np.array([[y0,y1]])]               
        :global_range: list([start, end])
        :returns: * the overlap code vector
                  * all the sub-intervals which are the union of every intervals list and global_range
                  * interval mask rergarding every sub-intervals
                  * interval weights
        """
        
        range_start, range_end = global_range
        global_range_array = np.array([global_range])
        
        # Get the timestamp array of each intervals list
        timestamps_list = [np.union1d(intervals, global_range_array) if intervals.size != 0 else global_range_array[0] for intervals in intervals_sequence ]

        # Compute the masks of complementary union with the global_interval 
        mask_list = [IntervalCompute.get_complementary_union(_, global_range) for _ in intervals_sequence]
        intervals_list = intervals_sequence + [global_range_array]

        # Get the union of all intervals boundaries values
        timestamps = reduce(np.union1d, [x.ravel() for x in intervals_list]) #TODO : reduce the ravelling?
        assert (range_start <= timestamps[0] and timestamps[-1] <= range_end), "some intervals are off range ([{},{}] instead of [{},{}])".format(timestamps[0], timestamps[-1], range_start, range_end)  

        # Convert the union of all timestamps back to intervals
        all_intervals = IntervalCompute.timestamps_to_intervals(timestamps, dtype = np.int64 if timestamps[0] < 0 else np.uint64)

        # Get their center
        x = all_intervals.mean(1)
        
        # Compute which interval is in which set
        interval_in_mask_list = [np.digitize(x, ts) - 1 for ts in timestamps_list]    
        all_interval_in_seq_list = [mask[inter_in] for mask, inter_in in zip(mask_list, interval_in_mask_list)]

        # Make the sum to get the confusion vector
        n = len(intervals_list) - 1 
        all_interval_in_seq_array = np.array(all_interval_in_seq_list) 
        weights = np.power(np.full(n,2), np.arange(n)).astype(np.uint32)
        confusion_vector = (all_interval_in_seq_array * weights[:,np.newaxis]).sum(0)

    #     print("mask_list = {}".format(mask_list))
    #     print("intervals_list = {}".format(intervals_list))
    #     print("timestamps = {}".format(timestamps))
    #     print("all_intervals = {}".format(all_intervals))

        # Printing formatted results
        if print_results:
            print("Intervals masks :")
            for inter_mask, w, k in zip(all_interval_in_seq_array,weights,["interval_{}".format(i) for i in range(len(intervals_sequence))]):
                print("{} {} ({})".format(inter_mask, w, k))
            print("{} <- confusion vector".format(confusion_vector))
        
        return confusion_vector, all_intervals, all_interval_in_seq_array, weights


    @staticmethod
    def display_operation_interval(Op_dict, Op_list, Op_colors = ["red", "blue"], color_mode="rainbow", 
                                   figsize=None, save_path=None):
        n_op = len(Op_list)
        # We copy the original dictionnary for modifications
        Op_dict_copy = Op_dict.copy()
        
        # Compute the max value and convert the intervals format ([f_start, f_end]) in ([f_start, f_width])
        max_intervals = 0
        for intervals in Op_dict.values():
            max_op_interval = intervals[-1][-1]
            if max_op_interval > max_intervals:
                max_intervals = max_op_interval
            intervals[:,1] -= intervals[:,0]
    
        # Figure creation
        fig, ax = plt.subplots(figsize=figsize)
        if "rainbow":
            colors = cm.rainbow(np.linspace(0, 1, n_op))
        else:
            colors = cycle(Op_colors)
        # Plotting data
        bar_ywitdh = 0.6
        bar_yHeightStep = 1
        frame_intervals_list = [Op_dict_copy[k] for k in Op_list]
        # List of the y-coordinate of the horizontal center of each row (ascending order)
    
        heigths_list = np.array([i*bar_yHeightStep for i in range(1,n_op+1)])
        # broken_barh loop
        for op, frame_intervals, color, h in zip(Op_list, frame_intervals_list, colors, heigths_list[::-1]):
            ax.broken_barh(frame_intervals, (h-(bar_ywitdh/2),bar_ywitdh), 
                           facecolors=color, edgecolors='black',
                           linewidth=1, linestyle='solid',
                           hatch=None)
        # Axe settings
        ax.set_title("Frame Intervals Visualisation")
        ax.set_ylim(0, (heigths_list[-1] + heigths_list[0]))
        ax.set_xlim(-(max_intervals * 0.05), max_intervals * 1.05)
        ax.set_xlabel('Frames')
        ax.set_yticks(heigths_list)
        ax.set_yticklabels(Op_list[::-1])
        ax.grid(True)
        ax.xaxis.grid(False)
        if save_path:
            fig.savefig(save_path)
        plt.show()
    
        
    @staticmethod
    def display_confusion_scoring(ref_intervals, sys_intervals, global_interval, confusion_results=None, 
                                  figsize=None, save_path=None, colors_def = ["cyan", "blue"]):
        """
        :param ref_intervals: reference np.ndarray supposely sorted by the first column
        :param sys_intervals: system np.ndarray supposely sorted by the first column
        """
        # Compute the max value and convert the intervals format ([f_start, f_end]) in ([f_start, f_width])
    #     max_intervals = max(ref_intervals[-1][-1],sys_intervals[-1][-1])
        interval_range = global_interval[1] - global_interval[0]
        frame_intervals_list = []
        # List of the y-coordinate of the horizontal center of each row (ascending order)
        bar_yHeightStep = 1
        heigths_list = []
        colors = []
        
        
        if ref_intervals.size != 0:
            ref_intervals_s_w = ref_intervals.copy()
            ref_intervals_s_w[:,1] -= ref_intervals_s_w[:,0]
            frame_intervals_list.append(ref_intervals_s_w)
            heigths_list.append(3*bar_yHeightStep if confusion_results is not None else 2*bar_yHeightStep)
            colors.append(colors_def[0])
            
        if sys_intervals.size != 0:
            sys_intervals_s_w = sys_intervals.copy()
            sys_intervals_s_w[:,1] -= sys_intervals_s_w[:,0]
            frame_intervals_list.append(sys_intervals_s_w)
            heigths_list.append(2*bar_yHeightStep if confusion_results is not None else 1*bar_yHeightStep)
            colors.append(colors_def[1])
            
        y_label_list = ["Reference", "System_Output"]
        n_rows = 2
        if confusion_results:
            confusion_vector, all_intervals, confusion_mapping = confusion_results
            all_interval_s_w = all_intervals.copy()
            all_interval_s_w[:,1] -= all_interval_s_w[:,0]
            frame_intervals_list.append(all_interval_s_w)
            heigths_list.append(bar_yHeightStep)
            y_label_list.append("Confusion")
            colors.append([confusion_mapping[x][1] for x in confusion_vector])
            n_rows = 3
            
        # Figure creation
        fig, ax = plt.subplots(figsize=figsize)

        # Plotting data
        bar_ywitdh = 0.6
        edge_color = ['black', (0.3,0.3,0.3), None]
        # broken_barh loop
        for frame_intervals, color, h in zip(frame_intervals_list, colors, heigths_list):
            ax.broken_barh(frame_intervals, (h-(bar_ywitdh/2),bar_ywitdh), 
                           facecolors=color, edgecolors=edge_color[1],
                           linewidth=1, linestyle='solid',
                           hatch=None)
            

        # Axe settings
        ax.set_title("Confusion visualisation")
        ax.set_ylim(0, (n_rows+1)*bar_yHeightStep)
        ax.set_xlim(-(interval_range * 0.05), interval_range * 1.05)
        ax.set_xlabel('Frames')
        ax.set_yticks([i * bar_yHeightStep for i in range(1,n_rows+1)])
        ax.set_yticklabels(y_label_list[::-1])
        ax.grid(True)
        ax.xaxis.grid(False)
        ax.set_facecolor((0.92,0.92,0.92))
        if save_path:
            fig.savefig(save_path)
        plt.show()

    @staticmethod
    def broken_barh(intervals, y_position, height, color='gray', mode="start_end", fig=None, options={}):
        """Create a broken_barh glyph and adds it to a figure if fig is given,
        else returns the source and the glyph.
        :param intervals: numpy.array
        :Note:
        Usage: > broken_barh(intervals, 2, 0.6, fig=p, color=["#9933FF"],options=default_options)
            or > source, glyph = broken_barh(intervals, 2, 0.6, color=["#FF00FF"], options=default_options)
               > p.add_glyph(source, glyph)
        """
        from bokeh.models.glyphs import Quad
        from bokeh.models import ColumnDataSource
        assert (mode in ['start_end', 'start_width']), "Unknown mode"
        assert (isinstance(color, list)), "color has to be a list"
        
        nb_interval = intervals.shape[0]
        half_heigth = height/2
        if mode == "start_end": right = intervals[:,1]
        elif mode == "start_width": right = intervals.sum(1)
        if len(color) == 1 and nb_interval != 1: color = nb_interval * [color[0]]

        quad_parameters = {"bottom": [y_position - half_heigth for i in range(nb_interval)],
                           "top":[y_position + half_heigth for i in range(nb_interval)],
                           "left": intervals[:,0],
                           "right": right,
                           "fill_color": color}
        
        if fig is not None:
            # append any other options
            for opt_name, opt_values in options.items():
                quad_parameters[opt_name] = opt_values
            quad_parameters["fill_color"] = color
            fig.quad(**quad_parameters)
        else:
            glyph = Quad(bottom="bottom", top="top", left="left", right="right", fill_color="fill_color", **options)
            source = ColumnDataSource(quad_parameters)
            return source, glyph

    @staticmethod
    def display_confusion_bokeh(ref_intervals, sys_intervals, global_range, confusion_data=None, 
                                colors = [["#9933FF"], ["#FF00FF"]], plot_size = [900,250]):

        from bokeh.io import output_notebook
        from bokeh.plotting import figure, show, output_file
        from bokeh.models import Range1d
        from bokeh.models.tickers import FixedTicker
        output_notebook()

        y_positions = np.array([2,1])
        intervals_list = [ref_intervals, sys_intervals]
        interval_range = global_range[1] - global_range[0]
        x_range = Range1d(-(interval_range*0.05), interval_range*1.05)
        
        if confusion_data is not None:
            y_range = Range1d(0, 4)
            con_intervals, confusion_vector, confusion_mapping = confusion_data
            colors.append([confusion_mapping[x][1] for x in confusion_vector])
            intervals_list.append(con_intervals)
            y_positions = np.append(y_positions + 1, 1)
        else:
            y_range = Range1d(0, 3)
        
        height = 0.6
        default_options = {"line_color":"black", "line_width":1}
        p = figure(x_range=x_range, y_range=y_range, 
                   plot_width=plot_size[0], plot_height=plot_size[1],
                   title="Confusion Visualisation",title_location='above')
        for interval, y_pos, col in zip(intervals_list, y_positions, colors):
            IntervalCompute.broken_barh(interval, y_pos, height, fig = p, color = col, options=default_options)
        
        labels = ["Reference","System_Output","Confusion"]
        p.xgrid.grid_line_color = None
        p.xaxis.axis_label = "Frames"
        p.yaxis.ticker = FixedTicker(ticks=y_positions[::-1])
        p.yaxis.major_label_overrides = {str(i):label for i,label in zip(y_positions, labels)}
        p.background_fill_color = (234,234,234)
        p.border_fill_color = (255,255,255)
        show(p)