NUM_OUTPUT=$(seq 32 148 32)
KERNEL_SIZE=$(seq 3 6)
STRIDE=$(seq 1 3)


for n in NUM_OUTPUT; do
    for k in KERNEL_SIZE; do
        for s in STRIDE; do
            ./modify.py variable-file.txt n k s
        done
    done
done

TOOLS=../../build/tools
for file in $(ls param_dir); do
    GLOG_alsologtostderr=1 $TOOLS/train_net.bin $file
done

modify.py
