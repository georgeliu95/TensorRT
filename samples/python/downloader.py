#!/usr/bin/env python3
#
# Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import argparse
import errno
import hashlib
import logging
import os
import sys


logger = logging.getLogger('downloader')


class DataFile:
    """Holder of a data file."""
    def __init__(self, attr):
        self.attr = attr
        self.path = attr['path']
        self.url = attr['url']
        if 'checksum' not in attr:
            logger.warning("Checksum of %s not provided!", self.path)
        self.checksum = attr.get('checksum', None)

    def __str__(self):
        return str(self.attr)


class SampleData:
    """Holder of data files of an sample."""
    def __init__(self, attr):
        self.attr = attr
        self.sample = attr['sample']
        files = attr.get('files', None)
        self.files = [DataFile(f) for f in files]

    def __str__(self):
        return str(self.attr)


def _loadYAML(yaml_path):
    with open(yaml_path, 'rb') as f:
        import yaml
        y = yaml.load(f, yaml.FullLoader)
        return SampleData(y)


def _checkMD5(path, refMD5):
    md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
    return md5 == refMD5


def _createDirIfNeeded(path):
    the_dir = os.path.dirname(path)
    try:
        os.makedirs(the_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
             raise


def download(data_dir, yaml_path, overwrite=False):
    """Download the data files specified in YAML file to a directory.

    Return false if the downloaded file or the local copy (if not overwrite) has a different checksum.
    """
    sample_data = _loadYAML(yaml_path)
    logger.info("Downloading data for %s", sample_data.sample)

    def _downloadFile(path, url):
        logger.info("Downloading %s from %s", path, url)
        import requests
        r = requests.get(url, stream=True, timeout=5)
        size = int(r.headers.get('content-length', 0))
        from tqdm import tqdm
        progress_bar = tqdm(total=size, unit='iB', unit_scale=True)
        with open(path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=1024):
                progress_bar.update(len(chunk))
                fd.write(chunk)
        progress_bar.close()

    allGood = True
    for f in sample_data.files:
        fpath = os.path.join(data_dir, f.path)
        if os.path.exists(fpath):
            if _checkMD5(fpath, f.checksum):
                logger.info("Found local copy %s, skip downloading.", fpath)
                continue
            else:
                logger.warning("Local copy %s has a different checksum!", fpath)
                if overwrite:
                    logging.warning("Removing local copy %s", fpath)
                    os.remove(fpath)
                else:
                    allGood = False
                    continue
        _createDirIfNeeded(fpath)
        _downloadFile(fpath, f.url)
        if not _checkMD5(fpath, f.checksum):
            logger.error("The downloaded file %s has a different checksum!", fpath)
            allGood = False

    return allGood


def _parseArgs():
    parser = argparse.ArgumentParser(description="Downloader of TensorRT sample data files.")
    parser.add_argument('-d', '--data', help="Specify the data directory, data will be downloaded to there. $TRT_DATA_DIR will be overwritten by this argument.")
    parser.add_argument('-f', '--file', help="Specify the path to the download.yml, default to `download.yml` in the working directory",
                        default='download.yml')
    parser.add_argument('-o', '--overwrite', help="Force to overwrite if MD5 check failed",
                        action='store_true', default=False)
    parser.add_argument('-v', '--verify', help="Verify if the data has been downloaded. Will not download if specified.",
                        action='store_true', default=False)

    args, _ = parser.parse_known_args()
    data = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if data is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    return data, args


def verifyChecksum(data_dir, yaml_path):
    """Verify the checksum of the files described by the YAML.

    Return false of any of the file doesn't existed or checksum is different with the YAML.
    """
    sample_data = _loadYAML(yaml_path)
    logger.info("Verifying data files and their MD5 for %s", sample_data.sample)

    allGood = True
    for f in sample_data.files:
        fpath = os.path.join(data_dir, f.path)
        if os.path.exists(fpath):
            if _checkMD5(fpath, f.checksum):
                logger.info("MD5 match for local copy %s", fpath)
            else:
                logger.error("Local file %s has a different checksum!", fpath)
                allGood = False
        else:
            allGood = False
            logger.error("Data file %s doesn't have a local copy", f.path)

    return allGood


def main():
    data, args = _parseArgs()
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    ret = True
    if args.verify:
        ret = verifyChecksum(data, args.file)
    else:
        ret = download(data, args.file, args.overwrite)

    if not ret:
        # Error of downloading or checksum
        sys.exit(1)


if __name__ == '__main__':
    main()


TRT_DATA_DIR = None

def getFilePath(path):
    """Util to get the full path to the downloaded data files.

    It only works when the sample doesn't have any other command line argument.
    """
    global TRT_DATA_DIR
    if not TRT_DATA_DIR:
        parser = argparse.ArgumentParser(description="Helper of data file download tool")
        parser.add_argument('-d', '--data', help="Specify the data directory where it is saved in. $TRT_DATA_DIR will be overwritten by this argument.")
        args, _ = parser.parse_known_args()
        TRT_DATA_DIR = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if TRT_DATA_DIR is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    fullpath = os.path.join(TRT_DATA_DIR, path)
    if not os.path.exists(fullpath):
        raise ValueError("Data file %s doesn't exist!" % fullpath)

    return fullpath
