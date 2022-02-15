from glob import glob
import os
import glob

for instrument in os.listdir("datasets/AIR/tfr48k/tst"):
    file_names = glob.glob("datasets/AIR/tfr48k/tst/"+instrument+"/*")
    for fn in file_names:
        fn2=fn.replace("datasets/AIR/tfr48k/tst/","datasets/AIR/tfr48k/dev/")
        if os.path.exists(fn2):
            print("removing")
            os.remove(fn2)

tst_fn = [f.replace("datasets/AIR/wav/tst/","") for f in glob.glob("datasets/AIR/wav/tst/"+instrument+"/*") ]
dev_fn = [f.replace("datasets/AIR/wav/dev/","") for f in glob.glob("datasets/AIR/wav/dev/"+instrument+"/*") ]

print(tst_fn)
print(dev_fn)
print(set(tst_fn).intersection(set(dev_fn)))