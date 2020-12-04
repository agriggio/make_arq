#!/usr/bin/env python
#
#    make_arq - A tool for generating Sony Pixel-Shift ARQ files
#    Copyright (C) 2018-2020 Alberto Griggio <alberto.griggio@gmail.com>
#
#    make_arq is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    make_arq is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function, division
import os, sys
import argparse
import tifffile
import numpy
import struct
import time
import tempfile
import json
import subprocess
import stat

try:
    import _makearq
    _has_makearq = True
except ImportError:
    _has_makearq = False


def color(row, col):
    return ((row & 1) << 1) + (col & 1)


def get_sony_frame_data(data, frame, idx, factor):
    filename, tags = frame
    width = tags['EXIF:ImageWidth']
    height = tags['EXIF:ImageHeight']
    offset = tags['EXIF:StripOffsets']
    rowstart = 1 if (idx // 4) >= 2 else 0
    colstart = 1 if (idx // 4) % 2 else 0
    r_off, c_off = {
        0 : (1, 1),
        1 : (0, 1),
        2 : (0, 0),
        3 : (1, 0)
        }[idx % 4]

    if _has_makearq:
        _makearq.get_sony_frame_data(data, filename, width, height, offset,
                                     factor, r_off, c_off, rowstart, colstart)
    else:
        fmt = '=' + ('H' * width)
        rowbytes = width * 2

        rowidx = list(range(height))
        colidx = list(range(width))

        with open(filename, 'rb') as f:
            f.seek(offset)
            for row in rowidx:
                d = f.read(rowbytes)
                v = struct.unpack(fmt, d)
                rr = (row + r_off) * factor + rowstart
                if rr >= 0:
                    rowdata = data[rr]
                    for col in colidx:
                        cc = (col + c_off) * factor + colstart
                        if cc >= 0:
                            c = color(row, col)
                            rowdata[cc][c] = v[col]        

try:
    import rawpy
    
    def get_fuji_frame_data(data, frame, idx, factor):
        filename = frame[0]
        
        with rawpy.imread(filename) as raw:
            im = raw.raw_image
            r_off, c_off = {
                0 : (0, 0),
                1 : (1, 0),
                2 : (0, 1),
                3 : (1, 1)
                }[idx % 4]
            if factor == 1:
                rowstart = 0
                colstart = 0
            else:
                rowstart, colstart = {
                    0 : (4, 2),
                    1 : (-1, 2),
                    2 : (4, -3),
                    3 : (-1, -3)
                    }[idx // 4]

            if _has_makearq:
                _makearq.get_fuji_frame_data(
                    im, len(im), len(im[0]), factor,
                    data, r_off, c_off, rowstart, colstart)
            else:
                rmax = len(im) * factor
                cmax = len(im[0]) * factor
                for y, row in enumerate(im):
                    rr = (y + r_off) * factor + rowstart
                    if rr >= 0 and rr < rmax:
                        rowdata = data[rr]
                        for x, v in enumerate(row):
                            cc = (x + c_off) * factor + colstart
                            if cc >= 0 and cc < cmax:
                                c = color(y, x)
                                rowdata[cc][c] = v
                            
except ImportError:
    def get_fuji_frame_data(*args):
        raise ValueError("please install rawpy to enable FUJIFILM support")
    

def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--force', action='store_true',
                   help='overwrite destination')
    p.add_argument('-o', '--output', help='output file')
    p.add_argument('-4', '--force-4', action='store_true',
                   help='force using 4 frames only, even if 16 are provided')
    p.add_argument('frames', nargs='+', help='the 4 (or 16) frames')
    opts = p.parse_args()
    if len(opts.frames) not in (4, 16):
        raise ValueError("please provide 4 or 16 frames (got %d)" %
                         len(opts.frames))
    return opts


def get_tags(filename):
    p = subprocess.Popen(['exiftool', '-json', '-b',
                          '-all:all', '-struct', '-G', filename],
                         stdout=subprocess.PIPE)
    out, _ = p.communicate()
    return json.loads(out)[0]


def check_valid_frames(frames):
    seq = set()
    width = set()
    height = set()
    make = set()
    model = set()
    lens = set()
    aperture = set()
    shutter = set()
    for (name, tags) in frames:
        seq.add(tags['MakerNotes:SequenceNumber'])
        make.add(tags['EXIF:Make'])
        model.add(tags['EXIF:Model'])
        lens.add(tags.get('EXIF:LensInfo'))
        aperture.add(tags.get('EXIF:FNumber'))
        shutter.add(tags.get('EXIF:ShutterSpeed'))
    if len(make) != 1 or make.pop() not in ('SONY', 'FUJIFILM'):
        raise ValueError('the frames must all come from a '
                         'SONY ILCE-7RM3, SONY ILCE-7RM4 '
                         'or FUJIFILM GFX 100 camera')
    if len(model) != 1 or model.pop() not in ('ILCE-7RM3', 'ILCE-7RM4',
                                              'GFX 100'):
        raise ValueError('the frames must all come from a '
                         'SONY ILC3-7RM3, ILCE-7RM4 '
                         'or FUJIFILM GFX 100 camera')
    is_sony = tags['EXIF:Make'] == 'SONY'
    if is_sony:
        width.add(tags['EXIF:ImageWidth'])
        height.add(tags['EXIF:ImageHeight'])
    else:
        width.add(tags['RAF:RawImageFullWidth'])
        height.add(tags['RAF:RawImageFullHeight'])        
    if len(width) != 1 or len(height) != 1:
        raise ValueError('the frames have different dimensions')
    if len(lens) != 1 or len(aperture) != 1 or len(shutter) != 1:
        raise ValueError('the frames have different lenses and/or exposures')
    off = min(seq)
    if seq != set(range(off, off+len(frames))):
        raise ValueError('the frames do not form a valid sequence')
    return is_sony


def get_frames(framenames):
    frames = []
    for name in framenames:
        tags = get_tags(name)
        frames.append((name, tags))
    is_sony = check_valid_frames(frames)
    seq2idx = {
        2 : 0,
        1 : 1,
        4 : 2,
        3 : 3,
        }
    def key(t):
        sn = t[1]['MakerNotes:SequenceNumber'] - 1
        s = 1 + (sn) % 4
        i = seq2idx[s]
        g = seq2idx[1 + sn // 4]
        return (g, i)
    frames.sort(key=key)
    if is_sony:
        w, h = frames[0][1]['EXIF:ImageWidth'], frames[0][1]['EXIF:ImageHeight']
    else:
        w, h = frames[0][1]['RAF:RawImageFullWidth'], \
               frames[0][1]['RAF:RawImageFullHeight']
    if len(frames) == 16:
        w *= 2
        h *= 2
    return frames, w, h, is_sony
    

def get_frame_data(data, frame, idx, is16, is_sony):
    if not is16:
        factor = 1
    else:
        factor = 2
    if is_sony:
        get_sony_frame_data(data, frame, idx, factor)
    else:
        get_fuji_frame_data(data, frame, idx, factor)


def write_pseudo_arq(filename, data, outtags):
    wb = None
    black = None
    is_sony = outtags['EXIF:Make'] == 'SONY'
    if is_sony:
        wbkey = "MakerNotes:WB_RGGBLevels"
        if wbkey in outtags:
            try:
                wb = [int(c) for c in outtags[wbkey].split()]
            except Exception as e:
                print("WARNING: can't determine camera WB (%s)" % str(e))
    else:
        wbkey = "RAF:WB_GRBLevels"
        if wbkey in outtags:
            try:
                wb = [int(c) for c in outtags[wbkey].split()]
                if len(wb) == 3:
                    wb.append(0)
            except Exception as e:
                print("WARNING: can't determine camera WB (%s)" % str(e))
        bkey = "RAF:BlackLevel"
        if bkey in outtags: 
            try:
                black = [int(c) for c in outtags[bkey].split()]
            except Exception as e:
                print("WARNING: can't determine black levels (%s)" % str(e))
           

    # set the WB where dcraw can find it
    extratags = []
    if wb is not None:
        extratags.append((29459, 'H', 4, wb))
    if black is not None:
        extratags.append((50714, 'H', 4, black))
    tifffile.imsave(filename, data, photometric=None, planarconfig='contig',
                    extratags=extratags)

    # try preserving the tags
    for key in ("SourceFile",
                "MakerNotes:SequenceNumber", 
                "EXIF:SamplesPerPixel",
                "EXIF:ImageWidth",
                "EXIF:ImageHeight",
                "EXIF:Compression",
                "EXIF:PhotometricInterpretation",
                "EXIF:SamplesPerPixel",
                "EXIF:PlanarConfiguration",
                "EXIF:StripOffsets",
                "EXIF:RowsPerStrip",
                "EXIF:StripByteCounts",
                "EXIF:ExifImageWidth",
                "EXIF:ExifImageHeight",
                "EXIF:FlashpixVersion",
                "EXIF:ColorSpace",
                "EXIF:Gamma",
                "EXIF:YCbCrCoefficients",
                "EXIF:YCbCrPositioning",
                ):
        if key in outtags:
            del outtags[key]
    if "EXIF:ImageDescription" not in outtags:
        outtags["EXIF:ImageDescription"] = ""
    outtags["EXIF:Software"] = "make_arq"
    for key in list(outtags.keys()):
        if key.startswith('MakerNotes:'):
            del outtags[key]
    fd, jsonname = tempfile.mkstemp('.json')
    os.close(fd)
    with open(jsonname, 'w') as out:
        json.dump([outtags], out)
    
    p = subprocess.Popen(['exiftool', '-overwrite_original',
                          #'-b',
                          '-G', '-j=' + jsonname, filename],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    err, _ = p.communicate()
    os.unlink(jsonname)
    
    # make sure the file is writable
    os.chmod(filename, 0o666)
    
    if p.returncode != 0:
        raise IOError(err)
    

def main():
    start = time.time()
    opts = getopts()
    if os.path.exists(opts.output) and not opts.force:
        raise IOError('output file "%s" already exists (use -f to overwrite)'
                      % opts.output)
    frames, width, height, is_sony = get_frames(opts.frames)
    is16 = len(frames) == 16
    if is16 and opts.force_4:
        frames = frames[:4]
        is16 = False
        width //= 2
        height //= 2
    data = numpy.zeros((height, width, 4), numpy.ushort)
    for idx, frame in enumerate(frames):
        print('Reading frame:', frame[0])
        get_frame_data(data, frame, idx, is16, is_sony)
    print('Writing combined data to %s...' % opts.output)
    write_pseudo_arq(opts.output, data, frames[0][1])
    end = time.time()
    print('Total time: %.3f' % (end - start))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR: %s' % str(e))
        exit(-1)
