import argparse
import sys
import numpy as np


from bundle_adjust import ba_timeseries
from bundle_adjust import data_loader


def main():
    
    
    parser = argparse.ArgumentParser(description='Bundle Adjustment for S2P')

    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing the necessary '
                              'parameters of the scene to be bundle adjusted.'))
    
    parser.add_argument('--timeline', action='store_true',                        
                        help=('only print the timeline of the scene described '
                              'by the configuration json.'))
    
    parser.add_argument('--verbose', action='store_true',                        
                        help=('print all information of the feature tracking '
                              'and bundle adjustment process.'))
    
    args = parser.parse_args()

    
    scene = ba_timeseries.Scene(args.config)
    timeline_indices = np.arange(len(scene.timeline)).tolist()
    
    if args.timeline:
        scene.get_timeline_attributes(timeline_indices, ['datetime', 'n_images', 'id'])
        sys.exit()
    
    # load options
    opt = data_loader.load_dict_from_json(args.config)

    # optional options
    opt['reconstruct'] = opt['reconstruct'] if 'reconstruct' in opt.keys() else False
    opt['pc3dr'] = opt['pc3dr'] if 'pc3dr' in opt.keys() else False
    opt['geotiff_label'] = opt['geotiff_label'] if 'geotiff_label' in opt.keys() else None

    # which timeline indices are to bundle adjust
    if 'timeline_indices' in opt.keys():
        timeline_indices = list(map(int,  opt['timeline_indices'].split(' ')))
        print('Found {} specific dates to adjust! timeline_indices: {}\n'.format(len(timeline_indices),
                                                                                 opt['timeline_indices']))
        scene.get_timeline_attributes(timeline_indices, ['datetime', 'n_images', 'id'])
    else:
        print('All dates will be adjusted!')

        
    # bundle adjust
    if opt['ba_method'] is None:
        print('\nSkipping bundle adjustment !\n')
    else:
        if opt['ba_method'] == 'ba_sequential':

            scene.run_sequential_bundle_adjustment(timeline_indices, n_previous=1, reset=True, verbose=args.verbose)

        elif opt['ba_method'] == 'ba_global':

            scene.run_global_bundle_adjustment(timeline_indices, reset=True, verbose=args.verbose)
        else:
            print('ba_method {} is not valid !'.format(opt['ba_method']))
            print('accepted values are: [ba_sequential, ba_global]')
            sys.exit()


    if opt['reconstruct']:
        scene.reconstruct_dates(timeline_indices, opt['ba_method'], std=True,
                                geotiff_label=opt['geotiff_label'], verbose=args.verbose)
            
    if opt['pc3dr']:
        scene.run_pc3dr(timeline_indices, opt['ba_method'])

    
if __name__ == "__main__":
    sys.exit(main()) 