#!/usr/bin/env python3
"""
Normalize the quarter-turn orientation of RPC GeoTIFF satellite images.

Given a directory of GeoTIFF images with RPC metadata, this script estimates the
image orientation induced by each RPC camera, finds the dominant orientation in
the set, and rotates the images whose orientation is inconsistent with that
mode. The rotation is restricted to multiples of 90 degrees and is applied both
to the raster pixels and to the RPC image-coordinate model.

The intended use is as a preprocessing step before feature extraction/matching,
so matchers such as LightGlue do not need pair-specific 90-degree corrections.
"""

import argparse
import glob
import os
import shutil
from collections import Counter

import numpy as np
import rasterio
import rpcm


RPC_COEFF_KEYS = {
    "LINE": ("LINE_NUM_COEFF", "LINE_DEN_COEFF"),
    "SAMP": ("SAMP_NUM_COEFF", "SAMP_DEN_COEFF"),
}


def get_id(fname):
    """Gets the basename without extension of a path to file."""
    return os.path.splitext(os.path.basename(fname))[0]


def parse_coeffs(v):
    return np.array([float(x) for x in str(v).replace(",", " ").split()], dtype=float)


def format_float(v):
    return "{:.17g}".format(float(v))


def format_coeffs(v):
    return " ".join(format_float(x) for x in np.asarray(v, dtype=float))


def snap_quarter_turns(angle_rad):
    """Return the nearest absolute orientation bin in quarter turns."""
    angle_deg = np.degrees(angle_rad) % 360.0
    return int(np.round(angle_deg / 90.0)) % 4


def rpc_center_and_scales(rpc):
    """Return RPC center and scale fields with rpcm-compatible names."""
    lon = getattr(rpc, "lon_offset", getattr(rpc, "lon0", 0.0))
    lat = getattr(rpc, "lat_offset", getattr(rpc, "lat0", 0.0))
    alt = getattr(rpc, "alt_offset", getattr(rpc, "height_offset", getattr(rpc, "h0", 0.0)))
    lat_scale = getattr(rpc, "lat_scale", 1.0)
    return float(lon), float(lat), float(alt), float(lat_scale)


def project_north_vector(rpc, frac=0.02):
    """
    Estimate the local pixel-space direction of geographic north.

    The returned points are in image coordinates (sample/column, line/row).
    """
    lon, lat, alt, lat_scale = rpc_center_and_scales(rpc)
    dlat = frac * lat_scale
    if not np.isfinite(dlat) or dlat == 0.0:
        dlat = 1e-6

    col0, row0 = rpc.projection(lon, lat, alt)
    col1, row1 = rpc.projection(lon, lat + dlat, alt)
    p0 = np.array([float(col0), float(row0)], dtype=float)
    p1 = np.array([float(col1), float(row1)], dtype=float)
    return p0, p1


def rotate_xy(points, width, height, k):
    """
    Rotate image coordinates using the same convention as np.rot90(array, k).

    Coordinates are (sample/column, line/row). k is the number of 90-degree
    counter-clockwise rotations applied to the raster.
    """
    pts = np.asarray(points, dtype=float)
    x = pts[..., 0]
    y = pts[..., 1]
    k = int(k) % 4

    if k == 0:
        xr, yr = x, y
    elif k == 1:
        xr, yr = y, (width - 1.0) - x
    elif k == 2:
        xr, yr = (width - 1.0) - x, (height - 1.0) - y
    elif k == 3:
        xr, yr = (height - 1.0) - y, x

    return np.stack([xr, yr], axis=-1)


def orientation_bin_after_rotation(rpc, width, height, k):
    p0, p1 = project_north_vector(rpc)
    q0, q1 = rotate_xy(np.vstack([p0, p1]), width, height, k)
    v = q1 - q0
    return snap_quarter_turns(np.arctan2(v[1], v[0]))


def estimate_orientation_bin(path):
    """Estimate the RPC-induced absolute orientation bin of one GeoTIFF."""
    rpc = rpcm.rpc_from_geotiff(path)
    p0, p1 = project_north_vector(rpc)
    v = p1 - p0
    k = snap_quarter_turns(np.arctan2(v[1], v[0]))
    return k, float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)


def coeff_channel(tags, channel):
    """Read one RPC output channel, either LINE or SAMP."""
    num_key, den_key = RPC_COEFF_KEYS[channel]
    return {
        "off": float(tags[f"{channel}_OFF"]),
        "scale": float(tags[f"{channel}_SCALE"]),
        "num": parse_coeffs(tags[num_key]),
        "den": parse_coeffs(tags[den_key]),
    }


def set_coeff_channel(tags, channel, ch):
    """Write one RPC output channel, either LINE or SAMP."""
    num_key, den_key = RPC_COEFF_KEYS[channel]
    tags[f"{channel}_OFF"] = format_float(ch["off"])
    tags[f"{channel}_SCALE"] = format_float(ch["scale"])
    tags[num_key] = format_coeffs(ch["num"])
    tags[den_key] = format_coeffs(ch["den"])


def affine_rpc_channel(old_ch, sign=1.0, const=0.0):
    """
    Construct a new RPC channel equal to sign * old_channel + const.

    Since quarter-turn rotations only swap LINE/SAMP and optionally negate them
    around an image-size-dependent constant, the RPC rational function can be
    updated exactly by reusing the original numerator/denominator coefficients.
    """
    sign = float(sign)
    const = float(const)
    if sign not in (-1.0, 1.0):
        raise ValueError("Only sign changes are supported")
    return {
        "off": const + sign * old_ch["off"],
        "scale": old_ch["scale"],
        "num": sign * old_ch["num"],
        "den": old_ch["den"].copy(),
    }


def rotate_rpc_tags(rpc_tags, width, height, k):
    """
    Rotate GDAL RPC metadata by k quarter turns counter-clockwise.

    RPC projection convention:
      SAMP = column coordinate, LINE = row coordinate.

    np.rot90 coordinate mappings from old (samp, line) to new coordinates are:
      k=1: samp' = line,          line' = width - 1 - samp
      k=2: samp' = width-1-samp,  line' = height - 1 - line
      k=3: samp' = height-1-line, line' = samp
    """
    k = int(k) % 4
    tags = dict(rpc_tags)
    old_line = coeff_channel(tags, "LINE")
    old_samp = coeff_channel(tags, "SAMP")

    if k == 0:
        new_samp = affine_rpc_channel(old_samp, +1, 0)
        new_line = affine_rpc_channel(old_line, +1, 0)
    elif k == 1:
        new_samp = affine_rpc_channel(old_line, +1, 0)
        new_line = affine_rpc_channel(old_samp, -1, width - 1.0)
    elif k == 2:
        new_samp = affine_rpc_channel(old_samp, -1, width - 1.0)
        new_line = affine_rpc_channel(old_line, -1, height - 1.0)
    elif k == 3:
        new_samp = affine_rpc_channel(old_line, -1, height - 1.0)
        new_line = affine_rpc_channel(old_samp, +1, 0)

    set_coeff_channel(tags, "SAMP", new_samp)
    set_coeff_channel(tags, "LINE", new_line)
    return tags


def copy_dataset_tags(src, dst):
    """Copy common dataset and band tags, excluding RPC tags handled separately."""
    dst.update_tags(**src.tags())
    for bidx in range(1, src.count + 1):
        dst.update_tags(bidx, **src.tags(bidx))
        if src.descriptions[bidx - 1] is not None:
            dst.set_band_description(bidx, src.descriptions[bidx - 1])


def write_rotated_geotiff(src_path, dst_path, k, overwrite=False, compress=None):
    """Write a rotated GeoTIFF and the corresponding rotated RPC metadata."""
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(dst_path)

    with rasterio.open(src_path) as src:
        rpc_tags = src.tags(ns="RPC")
        if not rpc_tags:
            raise RuntimeError("Missing RPC metadata in {}".format(src_path))

        width, height = src.width, src.height
        new_rpc_tags = rotate_rpc_tags(rpc_tags, width, height, k)

        if k == 0:
            # Re-write rather than symlink/copy so all outputs are uniform and live
            # in the requested directory. This also normalizes RPC tag formatting.
            data = src.read()
            out_height, out_width = height, width
        else:
            data = np.rot90(src.read(), k=k, axes=(1, 2))
            out_height, out_width = data.shape[1], data.shape[2]

        profile = src.profile.copy()
        profile.update(height=out_height, width=out_width)
        if compress is not None:
            profile.update(compress=compress)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)
            copy_dataset_tags(src, dst)
            dst.update_tags(ns="RPC", **new_rpc_tags)


def collect_tifs(input_dir, pattern):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    return [p for p in paths if os.path.isfile(p)]


def choose_rotation_to_match(path, target_bin):
    with rasterio.open(path) as src:
        rpc = rpcm.rpc_from_geotiff(path)
        width, height = src.width, src.height
        candidates = []
        for k in range(4):
            current_bin = orientation_bin_after_rotation(rpc, width, height, k)
            candidates.append((k, current_bin))
        good = [k for k, b in candidates if b == target_bin]
        if not good:
            raise RuntimeError(
                "Could not find a 90-degree rotation aligning {} to orientation bin {}. "
                "Candidates were {}".format(path, target_bin, candidates)
            )
        return min(good, key=lambda x: min(x, 4 - x)), candidates


def normalize_orientation(input_dir, output_dir, pattern="*.tif", overwrite=False, dry_run=False, compress=None):
    geotiff_paths = collect_tifs(input_dir, pattern)
    if len(geotiff_paths) == 0:
        raise RuntimeError("No files matching {} found in {}".format(pattern, input_dir))

    print("Found {} GeoTIFF images".format(len(geotiff_paths)))

    orientation_bins = []
    angles_deg = []
    for path in geotiff_paths:
        k, angle = estimate_orientation_bin(path)
        orientation_bins.append(k)
        angles_deg.append(angle)

    counts = Counter(orientation_bins)
    target_bin = counts.most_common(1)[0][0]
    print("Dominant RPC orientation bin: {} deg ({} / {} images)".format(90 * target_bin, counts[target_bin], len(geotiff_paths)))

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    for path, src_bin, angle in zip(geotiff_paths, orientation_bins, angles_deg):
        k_apply, candidates = choose_rotation_to_match(path, target_bin)
        dst_path = os.path.join(output_dir, os.path.basename(path))
        summary_rows.append((os.path.basename(path), angle, 90 * src_bin, k_apply, 90 * k_apply, dst_path))
        print(
            "{}: north_angle={:.2f} deg, bin={} deg, apply k={} ({} deg CCW)".format(
                os.path.basename(path), angle, 90 * src_bin, k_apply, 90 * k_apply
            )
        )
        if not dry_run:
            write_rotated_geotiff(path, dst_path, k_apply, overwrite=overwrite, compress=compress)

    summary_path = os.path.join(output_dir, "quarter_turn_orientation_summary.csv")
    if not dry_run:
        with open(summary_path, "w") as f:
            f.write("filename,north_angle_deg,src_orientation_bin_deg,k_apply,rotation_ccw_deg,output_path\n")
            for row in summary_rows:
                f.write("{},{:.8f},{},{},{},{}\n".format(*row))
        print("Wrote summary: {}".format(summary_path))


def main():
    parser = argparse.ArgumentParser(description="Normalize quarter-turn orientation of RPC GeoTIFF images.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .tif/.tiff images with RPC metadata")
    parser.add_argument("--output_dir", required=True, help="Directory where orientation-normalized GeoTIFFs are written")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern inside input_dir, e.g. '*.tif' or '*.TIF'")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--dry_run", action="store_true", help="Only estimate rotations; do not write images")
    parser.add_argument("--compress", default=None, help="Optional rasterio/GDAL compression, e.g. DEFLATE or LZW")
    args = parser.parse_args()

    normalize_orientation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
