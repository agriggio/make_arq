#!/usr/bin/env python
#
#    make_arq - A tool for generating Sony A7RIII Pixel-Shift ARQ files
#    Copyright (C) 2018 Alberto Griggio <alberto.griggio@gmail.com>
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
from __future__ import print_function
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
    from _makearq import get_frame_data as _get_frame_data
except ImportError:
    def _get_frame_data(data, filename, idx, width, height, offset,
                        factor=1, rowstart=0, colstart=0):
        fmt = '=' + ('H' * width)
        rowbytes = width * 2
        r_off, c_off = {
            0 : (1, 1),
            1 : (0, 1),
            2 : (0, 0),
            3 : (1, 0)
            }[idx]

        def color(row, col):
            return ((row & 1) << 1) + (col & 1)
        with open(filename, 'rb') as f:
            f.seek(offset)
            for row in xrange(height):
                d = f.read(rowbytes)
                v = struct.unpack(fmt, d)
                rr = (row + r_off - 1) * factor + rowstart
                if rr >= 0:
                    rowdata = data[rr]
                    for col in xrange(width):
                        cc = (col + c_off - 1) * factor + colstart
                        if cc >= 0:
                            c = color(row, col)
                            rowdata[cc][c] = v[col]        


def getopts():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--force', action='store_true',
                   help='overwrite destination')
    p.add_argument('-o', '--output', help='output file')
    p.add_argument('frames', nargs='+', help='the 4 (or 16) frames')
    opts = p.parse_args()
    if len(opts.frames) not in (4, 16):
        raise ValueError("please provide 4 or 16 frames (got %d)" % opts.frames)
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
        width.add(tags['EXIF:ImageWidth'])
        height.add(tags['EXIF:ImageHeight'])
        make.add(tags['EXIF:Make'])
        model.add(tags['EXIF:Model'])
        lens.add(tags.get('EXIF:LensInfo'))
        aperture.add(tags.get('EXIF:FNumber'))
        shutter.add(tags.get('EXIF:ShutterSpeed'))
    if len(make) != 1 or make.pop() != 'SONY':
        raise ValueError('the frames must all come from a '
                         'SONY ILCE-7RM3 or ILCE-7RM4 camera')
    if len(model) != 1 or model.pop() not in ('ILCE-7RM3', 'ILCE-7RM4'):
        raise ValueError('the frames must all come from a '
                         'SONY ILC3-7RM3 or ILCE-7RM4 camera')
    if len(width) != 1 or len(height) != 1:
        raise ValueError('the frames have different dimensions')
    if len(lens) != 1 or len(aperture) != 1 or len(shutter) != 1:
        raise ValueError('the frames have different lenses and/or exposures')
    off = min(seq)
    if seq != set(range(off, off+len(frames))):
        raise ValueError('the frames do not form a valid sequence')


def get_frames(framenames):
    frames = []
    for name in framenames:
        tags = get_tags(name)
        frames.append((name, tags))
    check_valid_frames(frames)
    # order according to the SequenceNumber tag
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
        g = seq2idx[1 + sn / 4]
        return (g, i)
    frames.sort(key=key)
    w, h = frames[0][1]['EXIF:ImageWidth'], frames[0][1]['EXIF:ImageHeight']
    if len(frames) == 16:
        w *= 2
        h *= 2
    return frames, w, h
    

def get_frame_data(data, frame, idx, is16):
    filename, tags = frame
    width = tags['EXIF:ImageWidth']
    height = tags['EXIF:ImageHeight']
    off = tags['EXIF:StripOffsets']
    if not is16:
        factor = 1
        rowstart = 0
        colstart = 0
    else:
        factor = 2
        rowstart = 1 if (idx / 4) >= 2 else 0
        colstart = 1 if (idx / 4) % 2 else 0
    _get_frame_data(data, filename, idx % 4, width, height, off,
                    factor, rowstart, colstart)


def write_pseudo_arq(filename, data, outtags):
    wb = None
    wbkey = "MakerNotes:WB_RGGBLevels"
    if wbkey in outtags:
        try:
            wb = [int(c) for c in outtags[wbkey].split()]
        except Exception as e:
            print("WARNING: can't determine camera WB (%s)" % str(e))

    # set the WB where dcraw can find it
    extratags = []
    if wb is not None:
        extratags.append((29459, 'H', 4, wb))
    tifffile.imsave(filename, data, photometric=None, planarconfig='contig',
                    extratags=extratags)

    # try preserving the tags
    for key in ('MakerNotes:SequenceNumber', 'SourceFile',
                'EXIF:SamplesPerPixel',
                'EXIF:ImageWidth', 'EXIF:ImageHeight'):
        if key in outtags:
            del outtags[key]
    fd, jsonname = tempfile.mkstemp('.json')
    os.close(fd)
    with open(jsonname, 'w') as out:
        json.dump([outtags], out)
    
    p = subprocess.Popen(['exiftool', '-overwrite_original',
                          '-b', '-G', '-j=' + jsonname, filename],
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
    frames, width, height = get_frames(opts.frames)
    data = numpy.empty((height, width, 4), numpy.ushort)
    is16 = len(frames) == 16
    for idx, frame in enumerate(frames):
        print('Reading frame:', frame[0])
        get_frame_data(data, frame, idx, is16)
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
