import numpy as np
import matplotlib.pyplot as plt


def plot_softl1_vs_linear_loss():

    plt.rcParams.update(plt.rcParamsDefault)
    params = {'backend': 'pgf',
                  'axes.labelsize': 22,
                  'font.size': 22,
                  'legend.fontsize': 22,
                  'xtick.labelsize': 22,
                  'ytick.labelsize': 22,
                  'text.usetex': False,  # use TeX for text
                  'font.family': 'serif',
                  'legend.loc': 'upper left',
                  'legend.fontsize': 22}
    plt.rcParams.update(params)
        
    soft_L1_loss = lambda z: 2 * ((1 + z)**0.5 - 1)
    x = np.linspace(-5, 5, 100)
    plt.figure(figsize=(20,5))
    plt.plot(x, soft_L1_loss(x**2))
    plt.plot(x, x**2)
    plt.legend(['soft_L1', 'squared L2'])
    plt.xlabel('reprojection distance')
    plt.ylabel('cost')
    plt.show()


def display_skysat_acquisition_scheme(r_h=5, overlap_ratio=0.2, save_pgf=False):
    import matplotlib.patches as patches
    from shapely.geometry import shape
    ar = 3199. / 1349.
    r_w = r_h * ar

    if save_pgf:
        params = {'backend': 'pgf',
                  'axes.labelsize': 22,
                  'ytick.labelleft': False,
                  'font.size': 22,
                  'legend.fontsize': 22,
                  'xtick.labelsize': 22,
                  'ytick.labelsize': 22,
                  'xtick.top': False,
                  'xtick.bottom': False,
                  'xtick.labelbottom': False,
                  'ytick.left': False,
                  'ytick.right': False,
                  'text.usetex': True,  # use TeX for text
                  'font.family': 'serif',
                  'legend.loc': 'upper left',
                  'legend.fontsize': 22}
        plt.rcParams.update(params)
    else:
        plt.rcParams.update(plt.rcParamsDefault)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    LW = 3

    # left strip
    rectangle = np.array([[0., 0.], [0., r_h], [r_w, r_h], [r_w, 0.]])
    for n in range(9):
        r_coords = rectangle.copy()
        r_coords[:, 1] += n * (1 - overlap_ratio) * r_h
        shapely_poly = shape({'type': 'Polygon', 'coordinates': [r_coords.tolist()]})
        ax.plot(*shapely_poly.exterior.xy, color='steelblue', linewidth=LW)

    # middle strip
    rectangle[:, 1] += 2 * (1 - overlap_ratio) * r_h
    rectangle[:, 0] += (1 - overlap_ratio / 2) * r_w
    for n in range(9):
        r_coords = rectangle.copy()
        r_coords[:, 1] += n * (1 - overlap_ratio) * r_h
        shapely_poly = shape({'type': 'Polygon', 'coordinates': [r_coords.tolist()]})
        ax.plot(*shapely_poly.exterior.xy, color='dodgerblue', linewidth=LW)

        # right strip
    rectangle = np.array([[0., 0.], [0., r_h], [r_w, r_h], [r_w, 0.]])
    rectangle[:, 0] += 2 * (1 - overlap_ratio / 2) * r_w
    for n in range(9):
        r_coords = rectangle.copy()
        r_coords[:, 1] += n * (1 - overlap_ratio) * r_h
        shapely_poly = shape({'type': 'Polygon', 'coordinates': [r_coords.tolist()]})
        ax.plot(*shapely_poly.exterior.xy, color='mediumblue', linewidth=LW)

    FS = 24 if save_pgf else 18

    # swath segment
    swath_w = rectangle[:, 0].max()
    x_values, y_values = [0, swath_w], [-3, -3]
    plt.plot(x_values, y_values, c='black')
    x_values, y_values = [0, 0], [-2, -4]
    plt.plot(x_values, y_values, c='black')
    x_values, y_values = [swath_w, swath_w], [-2, -4]
    plt.plot(x_values, y_values, c='black')
    ax.text(swath_w / 4, -5, 'swath width (6.6 km)', fontsize=FS)

    import matplotlib.patches as patches
    # image example rectangle
    for n in range(4):
        plt.plot([r_coords[n, 0], r_coords[n, 0] + r_w / 1.2],
                 [r_coords[n, 1], r_coords[n, 1] + r_h / 0.7], '--k', alpha=0.3)
    rect = patches.Rectangle((r_coords[0, 0] + r_w / 1.2, r_coords[0, 1] + r_h / 0.7), r_w, r_h,
                             linewidth=1, edgecolor='k', color='k', fill=True, alpha=0.15, zorder=2)
    ax.add_patch(rect)
    ax.text(r_coords[0, 0] + r_w / 1.2 + r_w / 22,
            r_coords[0, 1] + r_h / 0.7 + r_h / 6, 'single frame\n     image', fontsize=FS)

    ax.axis('off')
    plt.axis('equal')

    if save_pgf:
        plt.savefig('skysat_acquisition.png', pad_inches=0, bbox_inches='tight', dpi=200)

    plt.show()


def display_skysat_sensor_scheme(r_h=5, save_pgf=False):
    from shapely.geometry import shape

    if save_pgf:
        params = {'backend': 'pgf',
                  'axes.labelsize': 22,
                  'ytick.labelleft': False,
                  'font.size': 22,
                  'legend.fontsize': 22,
                  'xtick.labelsize': 22,
                  'ytick.labelsize': 22,
                  'xtick.top': False,
                  'xtick.bottom': False,
                  'xtick.labelbottom': False,
                  'ytick.left': False,
                  'ytick.right': False,
                  'text.usetex': True,  # use TeX for text
                  'font.family': 'serif',
                  'legend.loc': 'upper left',
                  'legend.fontsize': 22}
        plt.rcParams.update(params)
    else:
        plt.rcParams.update(plt.rcParamsDefault)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    r_h = 5
    r_w = r_h * 3199. / 1349. / 2
    FS = 18

    def draw_sensor(x, y, sensor_id):
        s3 = np.array([[x, y], [x, y + r_h], [x + r_w, y + r_h], [x + r_w, y]])
        r_coords = s3.copy()
        shapely_poly = shape({'type': 'Polygon', 'coordinates': [r_coords.tolist()]})
        ax.plot(*shapely_poly.exterior.xy, color='k')

        subband_h = r_h / 2 / 4
        for i in [1, 2, 3, 4]:
            plt.plot([x, x + r_w], [y + subband_h * i, y + subband_h * i], 'k')

        # add labels
        ax.text(x + r_w / 2.5, y + subband_h * 1 - subband_h + subband_h / 6, 'NIR', fontsize=FS)
        ax.text(x + r_w / 2.5, y + subband_h * 2 - subband_h + subband_h / 6, 'red', fontsize=FS)
        ax.text(x + r_w / 2.8, y + subband_h * 3 - subband_h + subband_h / 6, 'green', fontsize=FS)
        ax.text(x + r_w / 2.6, y + subband_h * 4 - subband_h + subband_h / 6, 'blue', fontsize=FS)
        ax.text(x + r_w / 2.6, y + subband_h * 6.5 - subband_h + subband_h / 6, 'PAN', fontsize=FS)
        ax.text(x + r_w / 3.4, y + subband_h * 9 - subband_h + subband_h / 2, 'sensor ' + sensor_id, fontsize=FS)

    # sensor 3
    x, y = 0., 0.
    draw_sensor(x, y, '3')
    # sensor 2
    x, y = r_w - 0.5, r_h + 0.5
    draw_sensor(x, y, '2')
    # sensor 1
    x, y = r_w - 0.5 + r_w - 0.5, 0
    draw_sensor(x, y, '1')
    # acquisition direction
    x, y = r_w - 0.5 + r_w / 2, r_h / 2
    plt.arrow(x, y, dx=0, dy=-r_h / 2 - r_h / 6, head_width=0.4, head_length=0.4, fc='k')
    ax.text(r_w - 0.5 + r_w / 4, -r_h / 2 - r_h / 20, 'acquisition\n  direction', fontsize=FS)

    ax.axis('off')
    plt.axis('equal')

    if save_pgf:
        plt.savefig('skysat_sensors.png', pad_inches=0, bbox_inches='tight', dpi=300)

    plt.show()


def display_row_of_dsms(dsms, vmin=None, vmax=None, c='cividis', aois=None,
                        custom_cb=True):

    if custom_cb:
        cb_frac, cb_pad, cb_asp = 0.045, 0.02, 7
        n_ticks, fontsize = 5, 22

    n_dsms = len(dsms)
    fig, axes = plt.subplots(1, n_dsms, figsize=(30,60))
    for i in range(n_dsms):
        if aois is not None:
            axes[i].plot(*aois[i].exterior.xy, color='black')
        current_vmin = np.nanmin(dsms[i]) if vmin is None else vmin
        current_vmax = np.nanmax(dsms[i]) if vmax is None else vmax
        im = axes[i].imshow(dsms[i], vmin=current_vmin, vmax=current_vmax, cmap=c)
        axes[i].axis('off')
        if custom_cb:
            delta = (current_vmax - current_vmin) / (n_ticks - 1)
            cb_tick_pos = [current_vmin + delta * k for k in range(n_ticks + 1)]
            if cb_tick_pos[-1] > current_vmax:
                cb_tick_pos[-1] = current_vmax
            cb_tick_labels = ['{:.2f}'.format(v) for v in cb_tick_pos]
            #if vmax is not None:
            #    cb_tick_labels[-1] = '> {:.2f}'.format(cb_tick_pos[-1])
            #if vmin is not None:
            #    cb_tick_labels[0] = '< {:.2f}'.format(cb_tick_pos[0])
            cb = fig.colorbar(im, ax=axes[i], fraction=cb_frac, pad=cb_pad,
                              aspect=cb_asp, ticks=cb_tick_pos)
            cb.ax.set_yticklabels(cb_tick_labels, fontsize=fontsize)
        else:
            cb = fig.colorbar(im, ax=axes[i])
    plt.show()


def get_GT_RBCT():

    # stocks (Mt)
    x = [3.70, 2.70, 2.70, 2.90, 3.13, 3.32, 3.31, 3.35, 3.54, 3.60, 3.84,
         3.85, 3.90, 3.66, 3.65, 3.50, 3.48, 4.37, 4.90, 4.60, 5.00, 5.50,
         5.37, 5.46, 5.42, 5.10, 5.10, 4.94, 4.90, 4.00, 4.10, 3.90]
    # dates
    dates_str = ['2019-12-19', '2020-01-08', '2020-01-14', '2020-01-23', '2020-02-07',
                 '2020-02-12', '2020-02-15', '2020-02-22', '2020-02-28', '2020-03-05',
                 '2020-03-10', '2020-03-11', '2020-03-18', '2020-03-24', '2020-03-26',
                 '2020-03-31', '2020-04-09', '2020-04-16', '2020-04-21', '2020-04-24',
                 '2020-04-28', '2020-05-01', '2020-05-06', '2020-05-12', '2020-05-19',
                 '2020-05-28', '2020-06-02', '2020-06-09', '2020-06-17', '2020-06-23',
                 '2020-06-26', '2020-06-30']
    return x, dates_str


def plot_RBCT_evolution_over_time(dsm_stock, dsm_labels, start_date='2020-01-14', end_date='2020-07-01'):

    import datetime
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    gt_stock, gt_labels = get_GT_RBCT()
    all_labels = np.unique(gt_labels + dsm_labels)
    all_dates = [datetime.datetime.strptime(l, '%Y-%m-%d') for l in all_labels]

    # preserve only dates between start_date and end_date
    date_dict = dict([(l, d) for l, d in zip(all_labels, all_dates) if d >= start_date and d <= end_date])
    all_labels = [l for l in all_labels if l in date_dict.keys()]
    all_dates = [datetime.datetime.strptime(l, '%Y-%m-%d') for l in all_labels]
    gt_stock = [gt_stock[i] for i in range(len(gt_stock)) if gt_labels[i] in date_dict.keys()]
    gt_labels = [l for l in gt_labels if l in date_dict.keys()]
    gt_labels_pos = [list(date_dict.keys()).index(l) for l in gt_labels]
    dsm_stock = [dsm_stock[i] for i in range(len(dsm_stock)) if dsm_labels[i] in date_dict.keys()]
    dsm_labels = [l for l in dsm_labels if l in date_dict.keys()]
    dsm_labels_pos = [list(date_dict.keys()).index(l) for l in dsm_labels]

    fontsize = 12
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.figure(figsize=(15, 5))
    plt.plot(gt_labels_pos, gt_stock, 'o')
    plt.plot(dsm_labels_pos, dsm_stock, 'o')
    plt.ylabel('stock (Mt)')
    plt.xlabel('time')
    n_dates = len(all_labels)
    date_label_freq = 1
    dates = [d.strftime("%d\n%b") for d in all_dates]
    print_freq = 2
    plt.xticks(np.arange(n_dates)[::print_freq], dates[::print_freq])
    legend_labels = ['ground truth', 'estimation']
    plt.legend(legend_labels, fontsize=fontsize)
    plt.show()


def compute_matches_over_time(scene, timeline_indices, out_dir):

    import timeit
    import copy
    from bundle_adjust import data_loader as loader

    def run_only_feature_tracking(scene, verbose=False):
        from bundle_adjust.camera_utils import get_perspective_optical_center
        from feature_tracks.ft_pipeline import FeatureTracksPipeline
        from bundle_adjust.ba_timeseries import suppress_stdout

        input_seq = [f['crop'] for f in scene.ba_data['crops']]
        offsets = [{k: c[k] for k in ['col0', 'row0', 'width', 'height']} for c in scene.ba_data['crops']]
        cameras, _ = loader.approx_perspective_projection_matrices(scene.ba_data['rpcs'], offsets, verbose=verbose)
        optical_centers = [get_perspective_optical_center(P) for P in cameras]
        footprints = get_image_footprints(scene.ba_data['rpcs'], offsets)
        local_data = {'n_adj': scene.ba_data['n_adj'], 'n_new': scene.ba_data['n_new'],
                      'fnames': scene.ba_data['image_fnames'], 'images': input_seq,
                      'rpcs': scene.ba_data['rpcs'], 'offsets': offsets, 'footprints': footprints,
                      'optical_centers': optical_centers, 'masks': scene.ba_data['masks']}

        with suppress_stdout():
            ft_pipeline = FeatureTracksPipeline(scene.ba_data['input_dir'], scene.ba_data['output_dir'],
                                                local_data, config=scene.tracks_config, satellite=True)
            feature_tracks = ft_pipeline.build_feature_tracks()
        return feature_tracks

    verbose = False
    scene = copy.copy(scene)
    scene.cam_model = 'perspective'
    ba_method = 'ba_global'
    ba_dir = os.path.join(scene.dst_dir, ba_method)
    os.makedirs(ba_dir, exist_ok=True)

    print('Computing matches over time...')
    matches_over_time = []
    counter = 0
    scene.tracks_config['continue'] = False
    for i in timeline_indices:

        t0 = timeit.default_timer()

        scene.tracks_config['predefined_pairs'] = None
        scene.set_ba_input_data([i], ba_dir, ba_dir, 0, verbose)
        feature_tracks = run_only_feature_tracking(scene, verbose=verbose)
        n_matches = len(feature_tracks['pairwise_matches']) if feature_tracks['C'] is not None else 0
        diff_days = 0.0
        matches_over_time.append((diff_days, n_matches))

        for j in np.array(timeline_indices)[np.array(timeline_indices) > i]:
            current_timeline_indices = [i, j]
            args = [scene.timeline, current_timeline_indices, 1, False]
            scene.tracks_config['predefined_pairs'] = loader.load_pairs_from_same_date_and_next_dates(*args)
            scene.set_ba_input_data(current_timeline_indices, ba_dir, ba_dir, 0, verbose)
            feature_tracks = run_only_feature_tracking(scene)
            n_matches = len(feature_tracks['pairwise_matches']) if feature_tracks['C'] is not None else 0
            d1, d2 = scene.timeline[i]['datetime'], scene.timeline[j]['datetime']
            diff_days = abs((d1 - d2).total_seconds() / (24.0 * 3600))
            matches_over_time.append((diff_days, n_matches))

        counter += 1
        running_time = timeit.default_timer() - t0
        print('{}/{} done in {:.2f} seconds'.format(counter, len(timeline_indices), running_time))

    np.savetxt(os.path.join(out_dir, 'matches_over_time.txt'), np.array(matches_over_time))
    np.savetxt(os.path.join(out_dir, 'matches_over_time_timeline_indices.txt'), np.array(timeline_indices))


def plot_matches_over_time(in_dir, scene, hist_consecutive_time_diff=True):

    matches_over_time = np.loadtxt(os.path.join(in_dir, 'matches_over_time.txt'))
    timeline_indices = np.loadtxt(os.path.join(in_dir, 'matches_over_time_timeline_indices.txt')).astype(int).tolist()

    max_days_diff = max(np.ceil(matches_over_time[:, 0]).astype(int))
    y = np.zeros(max_days_diff)
    y_counts = np.zeros(max_days_diff)

    for [diff_days, n_matches] in matches_over_time.tolist():
        idx = 0 if diff_days == 0 else np.ceil(diff_days).astype(int) - 1
        y[idx] += n_matches
        y_counts[idx] += 1

    # plot average number of matches over temporal distances in days
    y_counts[y_counts == 0] = 1
    y_avg = y / y_counts
    plt.bar(np.arange(max_days_diff), y_avg)
    plt.ylabel('average number of stereo matches')
    plt.xlabel('time distance (days)')
    plt.xticks(np.arange(0, 120, 10))
    plt.show()

    y_avg_merged = np.repeat([y_avg[i] + y_avg[i + 1] for i in np.arange(0, len(y_avg) - 1, 2)], 2)
    plt.bar(np.arange(max_days_diff), y_avg_merged)
    plt.ylabel('average number of stereo matches')
    plt.xlabel('time distance (days)')
    plt.xticks(np.arange(0, 120, 10))
    plt.show()

    if hist_consecutive_time_diff:
        scene.plot_timeline(timeline_indices)


def display_overlap_between_image_footprints(footprints, resolution):
    from bundle_adjust import geojson_utils
    from bundle_adjust import data_loader
    from shapely.geometry import shape

    n_footprints = len(footprints)

    utm_geojsons = []
    for f in footprints:
        coordinates = np.array(f['poly'].exterior.xy).T[:-1, :]
        utm_geojsons.append({'type': 'Polygon', 'coordinates': [(coordinates).tolist()]})

    boundary = geojson_utils.combine_utm_geojson_borders(utm_geojsons)

    boundary_coords = np.array(boundary['coordinates'][0])
    easts, norths = boundary_coords[:, 0], boundary_coords[:, 1]
    norths[norths < 0] += 10e6

    utm_bbx = {'xmin': easts.min(), 'xmax': easts.max(), 'ymin': norths.min(), 'ymax': norths.max()}

    height = int(np.floor((utm_bbx['ymax'] - utm_bbx['ymin']) / resolution) + 1)
    width = int(np.floor((utm_bbx['xmax'] - utm_bbx['xmin']) / resolution) + 1)

    mask_footprints = np.zeros((height, width, n_footprints), dtype=np.float32)

    for im_idx, x in enumerate(utm_geojsons):
        # create mask for current footprint

        coords = np.array(x['coordinates'][0])
        easts, norths = coords[:, 0], coords[:, 1]
        norths[norths < 0] += 10e6

        rows = (height - (norths - utm_bbx['ymin']) / resolution).astype(int)
        cols = ((easts - utm_bbx['xmin']) / resolution).astype(int)
        poly_verts_colrow = np.vstack([cols, rows]).T

        shapely_poly = shape({'type': 'Polygon', 'coordinates': [poly_verts_colrow.tolist()]})
        mask_footprints[:, :, im_idx] = data_loader.mask_from_shapely_polygons([shapely_poly], (height, width))

    image_counts = np.sum(mask_footprints, axis=2)
    image_counts[image_counts == 0] = np.nan

    plt.figure(figsize=(10, 10))
    im = plt.imshow(image_counts)
    ax = plt.gca()
    ax.axis('off')
    cb_frac, cb_pad, cb_asp = 0.045, 0.02, 7
    vmin = np.nanmin(image_counts)
    vmax = np.nanmax(image_counts)
    v_percentile = lambda x: np.ceil(np.nanpercentile(image_counts, x))
    n_ticks = 4
    delta = (vmax - vmin) / (n_ticks - 1)
    cb_tick_pos = [vmin + delta * k for k in range(n_ticks + 1)]
    cb = plt.colorbar(im, ax=ax, fraction=cb_frac, pad=cb_pad,
                      aspect=cb_asp, ticks=cb_tick_pos)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_ylabel('image count\n', rotation=270, fontsize=14)
    plt.show()



def plot_teaser_rpc_fitting():
    import rpcm
    from bundle_adjust import rpc_utils
    from bundle_adjust import data_loader as loader
    from s2p import geographiclib
    from bundle_adjust import camera_utils
    from bundle_adjust import ba_rotate
    import mpl_toolkits.mplot3d.art3d as art3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rpc_path = 'data/skysat_coal_singledate/20190617_075138_ssc2d2_0005_basic_panchromatic_dn_RPC.TXT'
    image_path = 'data/skysat_coal_singledate/20190617_075138_ssc2d2_0005_basic_panchromatic_dn.tif'

    rpc = rpcm.rpc_from_rpc_file(rpc_path)
    x, y = 0, 0
    h, w = loader.read_image_size(image_path)
    alt = 400  # rpc.alt_offset

    col_range, lin_range, alt_range = [x, x + w, 10], [y, y + h, 10], [alt - 100, alt + 100, 10]

    cols, lins, alts = rpc_utils.generate_point_mesh(col_range, lin_range, alt_range)
    lons, lats = rpc.localization(cols, lins, alts)
    x, y, z = geographiclib.lonlat_to_geocentric(lons, lats, alts)
    world_points = np.vstack([x, y, z]).T

    # world_points = ba_rotate.euler_angles_to_R(0.00002, 0.0001, 0.00005) @ world_points.T

    image_points = camera_utils.apply_rpc_projection(rpc, world_points)
    image_points[:, 0] += -500
    image_points[:, 1] += -500

    # plt.figure()
    # plt.scatter(image_points[:,0], image_points[:,1])
    # plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    rotated_tmp = ba_rotate.euler_angles_to_R(0.17 / 2, 0.17 / 2, -0.17 / 3) @ np.vstack([cols, lins, alts])
    rotated_tmp = rotated_tmp.T
    z_offset = 500
    ax.scatter(rotated_tmp[:, 0], rotated_tmp[:, 1], rotated_tmp[:, 2] + z_offset, marker='o')
    # ax.scatter(cols, lins, alts, marker='o')

    x_im = [image_points[:, 0].min(), image_points[:, 0].max(), image_points[:, 0].max(), image_points[:, 0].min()]
    y_im = [image_points[:, 1].min(), image_points[:, 1].min(), image_points[:, 1].max(), image_points[:, 1].max()]
    z_im = [0, 0, 0, 0]
    verts = [list(zip(x_im, y_im, z_im))]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='w', edgecolor='k', linewidths=1, alpha=0.0))

    ax.scatter(image_points[:, 0], image_points[:, 1], [0] * len(image_points), marker='x')
    x_min = min(image_points[:, 0].min(), rotated_tmp[:, 0].min())
    x_max = max(image_points[:, 0].max(), rotated_tmp[:, 0].max())
    y_min = min(image_points[:, 1].min(), rotated_tmp[:, 1].min())
    y_max = max(image_points[:, 1].max(), rotated_tmp[:, 1].max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, rotated_tmp[:, 2].max() + z_offset)
    ax.set_axis_off()
    plt.show()


def plot_timeline(scene, timeline_indices=None, filename=None, date_label_freq=2, fs=22):
    plt.rcParams.update(plt.rcParamsDefault)
    if filename is not None:
        params = {'backend': 'pgf',
                  'axes.labelsize': fs,
                  'ytick.labelleft': True,
                  'font.size': fs,
                  'legend.fontsize': fs,
                  'xtick.labelsize': fs,
                  'ytick.labelsize': fs,
                  'xtick.top': False,
                  'xtick.bottom': True,
                  'xtick.labelbottom': True,
                  'ytick.left': True,
                  'ytick.right': True,
                  'text.usetex': True,  # use TeX for text
                  'font.family': 'serif',
                  'legend.loc': 'upper right',
                  'legend.fontsize': fs}
        plt.rcParams.update(params)
    else:
        plt.rcParams.update(plt.rcParamsDefault)

    if timeline_indices is None:
        timeline_indices = np.arange(len(scene.timeline)).tolist()

    # plot distribution of temporal distances between consecutive dates
    # and plot also the number of images available per date

    n_dates = len(timeline_indices)
    dt2str = lambda t: t.strftime("%d %b\n%Hh")  # %b to get month abreviation
    dates = [dt2str(scene.timeline[timeline_indices[0]]['datetime'])]
    diff_in_days = []
    for i in range(n_dates - 1):
        d1 = scene.timeline[timeline_indices[i]]['datetime']
        d2 = scene.timeline[timeline_indices[i + 1]]['datetime']
        delta_days = abs((d1 - d2).total_seconds() / (24.0 * 3600))
        diff_in_days.append(delta_days)
        dates.append(dt2str(d2))

    n_ims = [scene.timeline[i]['n_images'] for i in timeline_indices]

    fontsize = fs
    plt.rcParams['xtick.labelsize'] = fontsize
    fig_w = 1 * n_dates / float(date_label_freq)
    if fig_w < 5:
        fig_w = fig_w * 2
    fig, ax1 = plt.subplots(figsize=(fig_w, 4))

    color = 'tab:blue'
    l1, = ax1.plot(np.arange(1, n_dates), diff_in_days, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=np.floor(min(diff_in_days)) - 0.2, top=np.ceil(max(diff_in_days)) + 0.7)
    ax1_yticks = np.arange(0, np.ceil(max(diff_in_days)) + 0.6, 2.5).astype(float)
    ax1.set_yticks(ax1_yticks)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    l2, = ax2.plot(np.arange(n_dates), n_ims, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=min(n_ims) - 1.2, top=max(n_ims) + 1.2)
    ax2_yticks = np.arange(min(n_ims) - 1, max(n_ims) + 2).astype(int)
    ax2.set_yticks(ax2_yticks)

    fontweight = 'bold'
    fontproperties = {'weight': fontweight, 'size': fontsize}
    ax1.set_yticklabels(['{:.1f}'.format(v) for v in ax1_yticks], fontproperties)
    ax2.set_yticklabels(ax2_yticks.astype(str).tolist(), fontproperties)

    plt.xticks(np.arange(n_dates)[::date_label_freq], np.array(dates)[::date_label_freq])
    legend_labels = ['distance to previous date in day units', 'number of images per date']
    plt.legend([l1, l2], legend_labels, fontsize=fontsize)
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    plt.show()