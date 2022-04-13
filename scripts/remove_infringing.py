from glob import glob
import os
import glob

disallowlist=['aGk1zIDQQjQ', 'GWU_50fQ6q0', 'SqWamN0ZYCo', 'z_I41tB8zSc', 'igCT4G1M1Xo', 'A8Q-O5KB7kw', 'LBZ1EvU5bYY', '2bYLzLsIWME', '_W2VM3uinZA', 'ur77HMoZPjY']

TARGET_DIR="datasets/AIR/tfr/tst/"

fps=glob.glob(f"{TARGET_DIR}/**/*",recursive=True)

for fp in fps:
    fn=fp.split("/")[-1]
    id=fn.split("_part")[0]
    if id in disallowlist:
        print(f"removing {fp}")
        os.remove(fp)

