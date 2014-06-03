#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_alsologtostderr=1 $TOOLS/train_net.bin cifar10_quick_solver_lane01.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
GLOG_alsologtostderr=1 $TOOLS/train_net.bin cifar10_quick_solver_lr1_lane01.prototxt
