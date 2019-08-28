#!/usr/bin/env sh
/home/user/caffe-spade/build/tools/caffe train --solver=train_solver.prototxt --gpu 3 2>&1| tee model.log
