import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors
from aind_ephys_utils import align
from pathlib import Path

#import analysis_funcs as af

MAX_WAVEFORM_CHANNELS = 150
WAVEFORM_SAMPLE_SLICE = slice(40, 160)


def _peak_channel_index(unit_waveform):
    magnitudes = np.abs(unit_waveform)
    finite_magnitudes = np.where(np.isfinite(magnitudes), magnitudes, -np.inf)
    if np.all(np.isneginf(finite_magnitudes)):
        return unit_waveform.shape[1] // 2
    return np.unravel_index(np.argmax(finite_magnitudes), unit_waveform.shape)[1]


def _unit_channel_ids(waveform_extractor, unit, num_template_channels):
    channel_ids = np.asarray(waveform_extractor.channel_ids)
    if len(channel_ids) == num_template_channels:
        return channel_ids

    sparsity = getattr(waveform_extractor, 'sparsity', None)
    if sparsity is not None:
        channel_indices = sparsity.unit_id_to_channel_indices[unit]
        sparse_channel_ids = channel_ids[channel_indices]
        if len(sparse_channel_ids) == num_template_channels:
            return sparse_channel_ids

    raise ValueError(
        f'Template for unit {unit} has {num_template_channels} channels, but '
        f'the waveform extractor provides {len(channel_ids)} channel IDs'
    )


def _waveform_channel_view(unit_waveform, channel_ids, max_channels=MAX_WAVEFORM_CHANNELS):
    unit_waveform = np.asarray(unit_waveform)
    channel_ids = np.asarray(channel_ids)
    if unit_waveform.ndim != 2:
        raise ValueError('Unit waveform must be a 2D samples-by-channels array')
    if unit_waveform.shape[1] != len(channel_ids):
        raise ValueError('Waveform channels and channel IDs must have equal lengths')
    if max_channels <= 0:
        raise ValueError('Maximum waveform channels must be positive')

    num_channels = unit_waveform.shape[1]
    if num_channels <= max_channels:
        return unit_waveform, channel_ids

    peak_channel = _peak_channel_index(unit_waveform)
    channel_start = int(np.clip(
        peak_channel - max_channels // 2,
        0,
        num_channels - max_channels,
    ))
    channel_stop = channel_start + max_channels
    return (
        unit_waveform[:, channel_start:channel_stop],
        channel_ids[channel_start:channel_stop],
    )


def _waveform_color_limits(unit_waveform):
    finite_values = unit_waveform[np.isfinite(unit_waveform)]
    if finite_values.size == 0:
        return -0.1, 0.1
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    voltage_scale = max(abs(min_value), abs(max_value))
    if voltage_scale == 0:
        return -0.1, 0.1
    zero_padding = max(voltage_scale * 1e-6, np.finfo(float).eps)
    min_value = min(-zero_padding, min_value)
    max_value = max(zero_padding, max_value)
    return min_value, max_value

def shiftedColorMap(cmap, min_val, max_val, name):
    '''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    Input
    -----
      cmap : The matplotlib colormap to be altered.
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.'''
    epsilon = 0.001
    start, stop = 0.0, 1.0
    min_val, max_val = min(0.0, min_val), max(0.0, max_val) # Edit #2
    try:
        midpoint = 1.0 - max_val/(max_val + abs(min_val))
    except(ZeroDivisionError):
        midpoint = 1.0
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    matplotlib.colormaps.register(cmap=newcmap, force=True)
    return newcmap


def raster_plot(event_locked_spike_times, time_range, cond_each_trial=None, raster=None, color='k', cond_colors = None, trial_start=0, ms=3, **kwargs):
    '''
    :param event_locked_spike_times: spike timestamps each trial relative to an event
    :param cond_each_trial: (OPTIONAL) some sort of label for each trial so that trials with the same parameters can be grouped together.
    :return: a cool raster plot
    '''
    if raster is None:
        raster = []

    if cond_each_trial is not None:
        conds = np.unique(cond_each_trial)

        if type(color) == str:
            color = np.tile(color, len(conds))
        if cond_colors is None:
            cond_colors = np.tile(['0.5', '0.75'], int(np.ceil(len(conds)/2)))

        total_trials = 0
        cond_lines = []
        cond_bars = []

        for indcond, cond in enumerate(conds):
            this_event_locked_spike_times = np.array(event_locked_spike_times, dtype=object)[cond_each_trial == cond]
            raster, none_cond_lines, none_cond_bars = raster_plot(this_event_locked_spike_times, time_range, raster=raster, color=color[indcond], trial_start=total_trials, ms=ms, **kwargs)
            total_trials += len(this_event_locked_spike_times)

            cond_line = plt.axhline(total_trials, color='0.7', zorder=-100)
            cond_lines.append(cond_line)

            xpos = [time_range[0]-0.03*(time_range[1]-time_range[0]),time_range[0]]
            ybot = [total_trials-len(this_event_locked_spike_times), total_trials-len(this_event_locked_spike_times)]
            ytop = [total_trials, total_trials]
            cond_bar = plt.fill_between(xpos, ybot, ytop,ec='none',fc=cond_colors[indcond], clip_on=False)
            cond_bars.append(cond_bar)


        trials_per_cond = total_trials/len(conds)
        plt.yticks(np.arange(trials_per_cond/2, total_trials, trials_per_cond), [f'{cond}' for cond in conds])
        plt.gca().tick_params('y', length=0, pad=8)

    else:
        Ntrials = len(event_locked_spike_times)
        for trial in range(Ntrials):
            this_raster = plt.plot(event_locked_spike_times[trial],
                                   (trial + 1 + trial_start) * np.ones(len(event_locked_spike_times[trial])),
                                   '.', color=color, rasterized=False, ms=ms, **kwargs)
            raster.append(this_raster)

        cond_lines = None
        cond_bars = None
        plt.ylim(0,Ntrials+2+trial_start)
        #zline = plt.axvline(0, color='0.8', zorder=-100)

    plt.xlim(time_range)
    return raster, cond_lines, cond_bars

def multi_unit_raster_plot(unit_ids, sorting_output, timestamps, waveform_extractor, event_ids, laser_onset_times, trial_types, probe, fig_title, segment=0, output_dir='/results'):
    '''
    Makes the output plots showing a summary of all the good tagged units
    '''
    width = np.ceil(np.sqrt(len(unit_ids)))
    height = np.ceil(len(unit_ids)/width)

    plt.clf()
    fig = plt.figure(figsize=((width * (len(trial_types)+1) * 3, height * 2)), constrained_layout=True)

    gs = gridspec.GridSpec(int(height), int(width), hspace=0.9, wspace=0.4, figure=fig)
    #gs.update(top=1-(height*0.01), bottom=0+(height*0.02), left=0+(width*0.015), right=1-(width*0.015), wspace=0.4, hspace=0.9)
    #gs.update(wspace=0.4, hspace=0.9)


    for ind_unit, unit in enumerate(unit_ids):
        sample_numbers = sorting_output.get_unit_spike_train(unit, segment_index=segment)
        #sample_numbers = sample_numbers[sample_numbers<len(timestamps)]
        unit_spike_times = timestamps[sample_numbers]
        #unit_waveform = waveform_extractor.get_unit_template(unit)
        template_ext = waveform_extractor.get_extension("templates")
        unit_waveform = template_ext.get_unit_template(unit)
        channel_ids = _unit_channel_ids(
            waveform_extractor, unit, unit_waveform.shape[1]
        )
        waveform_channel_view, displayed_channel_ids = _waveform_channel_view(
            unit_waveform, channel_ids
        )
        waveform_view = waveform_channel_view[WAVEFORM_SAMPLE_SLICE]
        #unit_metrics = laser_response_metrics.query('unit_id == @unit')

        gs_this_unit = gridspec.GridSpecFromSubplotSpec(2, len(trial_types)+1, subplot_spec=gs[int(ind_unit//width), int(ind_unit%width)], wspace=0.8, hspace=0.6, height_ratios=[0.005,1])
        #gs_this_cool_unit = gridspec.GridSpecFromSubplotSpec(2, len(width_ratios), subplot_spec=gs_cool_units[ind_unit//num_cols, np.mod(ind_unit, num_cols)], wspace=0.2, hspace=0.5, height_ratios=[0.005,1])

        for ind_type, trial_type in enumerate(trial_types):
            # plot npopto stim by site
            if 'internal' in trial_type:
                max_power = max(event_ids.query('type == @trial_type').power)
                sites = list(np.unique(event_ids.query('type == @trial_type').site))
                tag_trials = event_ids.query('param_group == "train" and site == @sites and power == @max_power and type == @trial_type and emission_location == @probe')
                y_axis = tag_trials.site.tolist()
                y_label = 'Emission site'
                y_ticks = sites
            elif 'external' in trial_type:
                tag_trials = event_ids.query('param_group == "train" and site == 0 and type == @trial_type and emission_location == @probe')
                powers = list(np.unique(event_ids.query('type == @trial_type').power))
                y_axis = tag_trials.power.tolist()
                y_label = 'Power (mW)'
                y_ticks = powers
            x_label = 'Time from laser onset (s)'

            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            total_duration = (duration*num_pulses)+(pulse_interval*num_pulses)
            raster_time_range = [-(total_duration/2)/1000, (1.5*total_duration)/1000]
            wavelength = np.unique(tag_trials.wavelength)[0]

            # plot responses to train of pulses
            ax_raster = plt.subplot(gs_this_unit[1 + ind_type//2, ind_type%2])
            this_event_timestamps = laser_onset_times[tag_trials.index.tolist()]
            #event_locked_timestamps = af.event_locked_timestamps(unit_spike_times, this_event_timestamps, raster_time_range)
            event_locked_timestamps, event_inds, unit_ids = align.to_events(unit_spike_times, this_event_timestamps, raster_time_range)
            # convert to the ragged array raster_plot needs
            ragged_array = []
            for indtrial in range(len(this_event_timestamps)):
                spikes_this_trial = event_locked_timestamps[event_inds==indtrial]
                ragged_array.append(spikes_this_trial)
            raster_plot(ragged_array, raster_time_range, y_axis, ms=2.5, markeredgecolor='none')
            #plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xlim(raster_time_range)
            ax_raster.set_yticklabels(y_ticks)
            plt.gca().tick_params('both', labelsize=8)
            plt.title(f'{wavelength} nm')

            # add xlabel to last units
            plt.xlabel("Time from laser onset (s)")


            # patches showing laser presentation
            yLims = np.array(plt.ylim())
            laser_color = 'tomato' if 'red' in trial_type else 'skyblue'
            for pulse in range(num_pulses):
                rect = patches.Rectangle((pulse * (duration+pulse_interval)/1000, yLims[0]), duration / 1000, yLims[1] - yLims[0], linewidth=1, edgecolor=laser_color, facecolor=laser_color, alpha=0.35, clip_on=False)
                ax_raster.add_patch(rect)

        # plot waveform
        ax_waveform = plt.subplot(gs_this_unit[1,len(trial_types)])
        cmap = matplotlib.cm.PRGn
        #cmap = matplotlib.cm.viridis
        min_voltage, max_voltage = _waveform_color_limits(waveform_view)
        voltage_norm = matplotlib.colors.TwoSlopeNorm(
            vmin=min_voltage,
            vcenter=0,
            vmax=max_voltage,
        )
        waveform_image = plt.imshow(
            waveform_view.T,
            aspect='auto',
            cmap=cmap,
            norm=voltage_norm,
        )
        ax_waveform.invert_yaxis()
        tick_count = min(4, len(displayed_channel_ids))
        channel_ticks = np.unique(np.linspace(
            0, len(displayed_channel_ids) - 1, tick_count, dtype=int
        ))
        ax_waveform.set_yticks(channel_ticks)
        ax_waveform.set_yticklabels(displayed_channel_ids[channel_ticks])
        plt.ylabel('Channel')
        # add xlabel to last units
        plt.xlabel('Sample number')
        cbar = plt.colorbar(waveform_image, ax=ax_waveform)
        cbar.set_label('Voltage (uV)')

        # inset with peak channel waveform
        peak_channel = _peak_channel_index(unit_waveform)
        peak_waveform = unit_waveform[WAVEFORM_SAMPLE_SLICE, peak_channel]

        ax_inset = inset_axes(ax_waveform, width="30%", height="25%", loc=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_waveform.transAxes)
        plt.plot(peak_waveform, lw=2, c='k')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        # Add ghost axes so I can add a title with cluster number...
        ax_title = plt.subplot(gs_this_unit[0, :])
        ax_title.axis('off')
        ax_title.set_title(f'cluster {unit}', fontweight='heavy')

    #gs.constrained_layout(plt.gcf())
    #height_scale = len(y_ticks)/4
    #plt.gcf().set_size_inches((width * (len(trial_types)+1) * 3, height * height_scale), constrained_layout=True)
    fig_format = 'png'
    fig_name = f'{fig_title}.{fig_format}'
    output_path = Path(output_dir) / fig_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=fig_format)
    print(f'{fig_name} saved')
    plt.close(fig)
    return output_path
def pulse_plot(unit_spike_times, trial_laser_onset_times, time_range, duration, inter_pulse_interval, num_pulses, latency, title=None, color='tomato'):
    for pulse in range(num_pulses):
        this_time_range = time_range + (pulse * inter_pulse_interval)
        this_pulse_latency_locked_timestamps, latency_event_ids, unneeded_ids = align.to_events(unit_spike_times, trial_laser_onset_times, this_time_range)
        plt.plot(this_pulse_latency_locked_timestamps - (pulse * inter_pulse_interval), latency_event_ids + (len(trial_laser_onset_times)*pulse), 'k.', ms=2)
        plt.axhline(len(trial_laser_onset_times)*pulse, color='0.75')
    plt.axvline(latency, color='blue', ls='--')
    plt.axvspan(0, duration, color=color, alpha=0.3)
    plt.xlim(time_range)
    plt.ylim(0, len(trial_laser_onset_times)*num_pulses)
    plt.yticks(range(int(len(trial_laser_onset_times)/2), int(len(trial_laser_onset_times)*(num_pulses+1)-(len(trial_laser_onset_times)/2)), int(len(trial_laser_onset_times))))
    plt.gca().set_yticklabels(range(1,num_pulses+1))
    #plt.ylabel('Pulse')
    #plt.xlabel('Time from laser onset (s)')
    if title is not None:
        plt.title(title)

def multi_unit_pulse_plot(unit_ids, sorting_output, timestamps, laser_response_metrics, laser_onset_times, trial_ids, trial_types, probe_name, fig_title, output_dir='/results'):
    width = np.ceil(np.sqrt(len(unit_ids)))
    height = np.ceil(len(unit_ids)/width)

    plt.clf()
    fig = plt.figure(figsize=((width*2*len(trial_types), height*3)), constrained_layout=True)

    gs = fig.add_gridspec(int(height), int(width), hspace=0.9, wspace=0.4, figure=fig)

    for ind_unit, unit_id in enumerate(unit_ids):
        sample_numbers = sorting_output.get_unit_spike_train(unit_id, segment_index=0)
        unit_spike_times = timestamps[sample_numbers]
        unit_metrics = laser_response_metrics.loc[laser_response_metrics['unit_id'] == unit_id]

        #gs_this_unit = gridspec.GridSpecFromSubplotSpec(2, len(trial_types), subplot_spec=gs[int(ind_unit//width), int(ind_unit%width)], wspace=0.7, hspace=0.6, height_ratios=[0.005,1])
        gs_this_unit = gs[int(ind_unit//width), int(ind_unit%width)].subgridspec(2, len(trial_types), wspace=0.6, height_ratios=[0.005,1])
        for ind_type, trial_type in enumerate(trial_types):
            best_power = float(unit_metrics[f'{trial_type}_train_best_power'].iloc[0])
            if np.isnan(best_power):
                this_type_trials = trial_ids[trial_ids['type']==trial_type]
                best_power = float(max(np.unique(this_type_trials['power'])))
            latency = float(unit_metrics[f'{trial_type}_train_best_mean_latency'].iloc[0])
            tag_trials = trial_ids.query(f'param_group == "train" and site == 0 and type == @trial_type and emission_location == @probe_name and power == @best_power')
            trial_laser_onset_times = laser_onset_times[tag_trials.index.tolist()]

            duration = np.unique(tag_trials.duration)[0]
            num_pulses = np.unique(tag_trials.num_pulses)[0]
            pulse_interval = np.unique(tag_trials.pulse_interval)[0]
            time_range = [-(duration/2)/1000, (duration*1.5)/1000]

            ax_raster = fig.add_subplot(gs_this_unit[1, ind_type])
            pulse_plot(unit_spike_times, trial_laser_onset_times, time_range, duration/1000, (duration + pulse_interval)/1000, num_pulses, latency, title=f'{best_power} mW', color='tomato' if 'red' in trial_type else 'skyblue')
            if ind_type == 0:
                plt.ylabel("Pulse")

        # group axes
        ax_group = fig.add_subplot(gs_this_unit[1,:])
        ax_group.set_xticks([])
        ax_group.set_yticks([])
        ax_group.set_frame_on(False)
        ax_group.set_xlabel("Time from laser onset (s)", labelpad=20)

        # Add ghost axes so I can add a title with cluster number...
        ax_title = plt.subplot(gs_this_unit[0, :])
        ax_title.axis('off')
        ax_title.set_title(f'cluster {unit_id}', fontweight='heavy')

    fig_format = 'png'
    fig_name = f'{fig_title}.{fig_format}'
    output_path = Path(output_dir) / fig_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=fig_format)
    print(f'{fig_name} saved')
    plt.close(fig)
    return output_path
