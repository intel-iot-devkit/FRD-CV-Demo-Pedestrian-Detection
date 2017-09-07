#!/usr/bin/env python3
import concurrent.futures
import argparse
import os.path
import struct
import csv
import cv2
import numpy

args = argparse.ArgumentParser("Utility to build BRISK object databases")
args.add_argument("csv", nargs=1, type=str,
        help="CSV file mapping image files to object names")
args.add_argument("objects", nargs=1, type=str,
        help="Base directory for object image files")
args.add_argument("-j", "--threads", type=int, default=None,
        help="Specify number of concurrent worker threads")
args.add_argument("-o", "--output", type=argparse.FileType(mode='wb'),
        help="Output to a specific file")
args = args.parse_args()
filename = args.csv[0]

try:
    args.csv = open(args.csv[0])
except IOError:
    print("Failed to open CSV file: {}".format(args.csv))
    exit(1)
args.objects = args.objects[0]

worker = concurrent.futures.ThreadPoolExecutor(max_workers=args.threads)

# Read CSV data and build descriptors
db = dict([(l[0],l[1:]) for l in csv.reader(args.csv)])
images = dict(worker.map(lambda x: (tuple(x[1]),
    cv2.cvtColor(cv2.imread(os.path.join(args.objects, x[0])), cv2.COLOR_BGR2GRAY)),
    db.items()))
descs = dict(worker.map(lambda r: (r[0],cv2.BRISK_create(30,3,1.0).detectAndCompute(r[1], None)),
        images.items()))

#k = list(images.keys())[0]
#print(k)
#mat = images[k].copy()
#mat2 = mat.copy()
#mat2 = cv2.drawKeypoints(mat, descs[k][0], mat2)
#cv2.imwrite("temp.png", mat2)
#
#exit(0)

# Serialize results into DB file
outfile = args.output
if not outfile:
    ofname = filename.replace(".csv", ".db")
    try:
        outfile = open(ofname, 'wb')
    except IOError:
        print("Failed to open output file: {}".format(ofname))

# DB file is an unbounded array of elements, each of which has the form
# db_entry {
#   u16be       title_len
#   utf8        title[title_len]
#   u16be       n_keypoints
#   keypoint    keypoints[n_keypoints]
#   u16be       n_descriptors
#   u8          descriptors[n_descriptors*n_keypoints] }
# keypoint {
#   f32be   x
#   f32be   y
# }
#
# NOTE: descriptor matrix is stored in row-major (C-style) order

numpy.set_printoptions(threshold=numpy.nan)
def write_entry(f, title, keypoints, descriptors):
    bin_title = title.encode("utf8")
    f.write(struct.pack(">H", len(bin_title)) + bin_title)

    #print(title, len(keypoints))
    f.write(struct.pack(">H", len(keypoints)))
    for k in keypoints:
        f.write(struct.pack(">ff", *k.pt))

    f.write(struct.pack(">H", descriptors.shape[1]))
    f.write(descriptors.tobytes(order='C'))
    #print(descriptors)

with outfile as f:
    for title, info in descs.items():
        write_entry(f, ':'.join(title), *info)
