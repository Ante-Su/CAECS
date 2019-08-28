#!/usr/bin/env sh
set -e
/home/user/caffe-spade/build/tools/caffe test -model=test.prototxt -weights=snapshot_no__iter_35000.caffemodel -iterations=100 -gpu=2  
