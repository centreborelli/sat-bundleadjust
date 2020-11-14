import argparse
import sys
import os
import numpy as np


from bundle_adjust import ba_timeseries
from bundle_adjust import data_loader


def main():

    parser = argparse.ArgumentParser(description='Bundle Adjustment for S2P')

    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing the necessary '
                              'parameters of the scene to be bundle adjusted.'))
    
    parser.add_argument('--timeline', action='store_true',                        
                        help=('just print the timeline of the scene described '
                              'by the configuration json, do not run anything else.'))
    
    parser.add_argument('--verbose', action='store_true',                        
                        help=('print all information of the feature tracking, '
                              'bundle adjustment and 3D reconstruction tasks.'))

    parser.add_argument('--reset', action='store_true',
                        help=('delete previous files '
                              'and run all from zero.'))
    
    # parse command line arguments
    args = parser.parse_args()

    # load options from config file and copy config file to output_dir
    opt = data_loader.load_dict_from_json(args.config)
    os.makedirs(opt['output_dir'], exist_ok=True)
    os.system('cp {} {}'.format(args.config, os.path.join(opt['output_dir'], os.path.basename(args.config))))
    
    # redirect all prints to a bundle adjustment logfile inside the output directory
    log_file = open('{}/{}_BA.log'.format(opt['output_dir'], data_loader.get_id(args.config)), 'w+')
    sys.stdout = log_file
    sys.stderr = log_file

    # load scene
    scene = ba_timeseries.Scene(args.config)
    timeline_indices = np.arange(len(scene.timeline), dtype=int).tolist()
    
    if args.timeline:
        scene.get_timeline_attributes(timeline_indices, ['datetime', 'n_images', 'id'])
        sys.exit()

    # optional options
    opt['reconstruct'] = opt['reconstruct'] if 'reconstruct' in opt.keys() else False
    opt['geotiff_label'] = opt['geotiff_label'] if 'geotiff_label' in opt.keys() else None
    opt['postprocess'] = opt['postprocess'] if 'postprocess' in opt.keys() else False
    opt['skip_ba'] = opt['skip_ba'] if 'skip_ba' in opt.keys() else False
    opt['s2p_parallel'] = opt['s2p_parallel'] if 's2p_parallel' in opt.keys() else 5
    opt['n_dates'] = opt['n_dates'] if 'n_dates' in opt.keys() else 1
    scene.tracks_config['max_kp'] = opt['max_kp'] if 'max_kp' in opt.keys() else 3000

    # which timeline indices are to bundle adjust
    if 'timeline_indices' in opt.keys():
        timeline_indices = [int(idx) for idx in opt['timeline_indices']]
        print_args = [len(timeline_indices), timeline_indices]
        print('Found {} selected dates ! timeline_indices: {}\n'.format(*print_args), flush=True)
        scene.get_timeline_attributes(timeline_indices, ['datetime', 'n_images', 'id'])
    else:
        print('All dates selected !\n', flush=True)

    # bundle adjust
    if opt['ba_method'] is None or opt['skip_ba']:
        print('\nSkipping bundle adjustment !\n')
    else:
        if opt['ba_method'] == 'ba_sequential':
            scene.run_sequential_bundle_adjustment(timeline_indices, previous_dates=opt['n_dates'],
                                                   reset=args.reset, verbose=args.verbose)
        elif opt['ba_method'] == 'ba_global':
            scene.run_global_bundle_adjustment(timeline_indices, next_dates=opt['n_dates'],
                                               reset=args.reset, verbose=args.verbose)
        elif opt['ba_method'] == 'ba_bruteforce':
            scene.run_bruteforce_bundle_adjustment(timeline_indices, reset=args.reset, verbose=args.verbose)
        else:
            print('ba_method {} is not valid !'.format(opt['ba_method']))
            print('accepted values are: [ba_sequential, ba_global, ba_bruteforce]')
            sys.exit()

    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()

    # reconstruct scene if specified and compute metrics
    if opt['reconstruct']:

        # redirect all prints to a reconstruct logfile inside the output directory
        log_file = open('{}/{}_3D.log'.format(opt['output_dir'], data_loader.get_id(args.config)), 'w+')
        sys.stdout = log_file
        sys.stderr = log_file

        scene.reconstruct_dates(timeline_indices, ba_method=opt['ba_method'], n_s2p=opt['s2p_parallel'],
                                geotiff_label=opt['geotiff_label'])
        if (opt['ba_method'] is not None) and opt['postprocess']:
            scene.project_pts3d_adj_onto_dsms(timeline_indices, opt['ba_method'])
            scene.interpolate_small_holes(timeline_indices, imscript_bin_dir='bin',
                                          ba_method=opt['ba_method'], geotiff_label=opt['geotiff_label'])
            scene.compute_stat_per_date(timeline_indices, ba_method=opt['ba_method'], stat='avg', use_cdsms=True,
                                        geotiff_label=opt['geotiff_label'], clean_tmp_warps=False)
            scene.compute_stat_per_date(timeline_indices, ba_method=opt['ba_method'], stat='std', use_cdsms=True,
                                        geotiff_label=opt['geotiff_label'])

        if opt['ba_method'] is None:
            scene.run_pc3dr_datewise(timeline_indices, ba_method=opt['ba_method'])
            scene.compute_stats_over_time(timeline_indices, ba_method=opt['ba_method'], pc3dr=True)
            scene.compute_dsm_registration_metrics(timeline_indices, ba_method=opt['ba_method'], pc3dr=True)

        if opt['ba_method'] is not None:
            scene.compute_stats_over_time(timeline_indices, ba_method=opt['ba_method'], pc3dr=False)
            scene.compute_dsm_registration_metrics(timeline_indices, ba_method=opt['ba_method'],
                                                   use_cdsms=opt['postprocess'])

        # close logfile
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())
