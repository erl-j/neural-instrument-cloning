from fileinput import filename
from locale import normalize
import pandas as pd
import json 
import numpy as np
import random
import librosa
import soundfile as sf
import os

TARGET_SR=44100
# read timestamps
df  = pd.read_csv('datasets/timestamps.csv')

INSTRUMENTS = ["Flute","Oboe","Clarinet","Bassoon","Saxophone","Trumpet","Horn","Trombone","Tuba"]

entries = []
for df_index, df_row in df.iterrows():
    #print(df_row)
    #print(df_row["label"])
    if df_row["label"] is not np.NaN:
        entries.append({"filename": df_row["audio"].replace("http://localhost:8081/", ""),"events" : json.loads(df_row["label"])})


print(f"n recordings before cleanup {len(entries)}")


# remove all files with "borderline" sections
entries = [e for e in entries if "Borderline" not in [event["labels"][0] for event in e["events"]]]


print(f"n recordings after cleanup {len(entries)}")
# for each instrument
# split the remaining tracks saving 4 tracks for testing

TEST_MIN=64
N_TEST=4

random.seed(1)

for instrument in INSTRUMENTS:

    instrument_entries = [e for e in entries if e["filename"].lower().startswith(instrument.lower())]
    instrument_entries = random.sample(instrument_entries, len(instrument_entries))
    viable_for_testing = [e for e in instrument_entries if True in [(ev["end"]-ev["start"])>=TEST_MIN for ev in e["events"]]]
    not_viable_for_testing = [e for e in instrument_entries if True not in [(ev["end"]-ev["start"])>=TEST_MIN for ev in e["events"]]]
    assert len(viable_for_testing)>=N_TEST
    test_entries=viable_for_testing[:N_TEST]
    dev_entries = not_viable_for_testing+viable_for_testing[N_TEST:]

    print(f"for {instrument} we have {len(test_entries)+len(dev_entries)}, {len(instrument_entries)} total entries, {len(test_entries)} test entries and {len(dev_entries)} dev entries")


    for split in ["dev","test"]:
        if split=="dev":
            split_entries=dev_entries
        else:
            split_entries=test_entries
        for entry in split_entries:
            y,sr = librosa.load(f"datasets/solos/wav/{entry['filename']}",sr=TARGET_SR)

            for event_idx,event in enumerate(entry["events"]):
                start=event["start"]
                end=event["end"]
   
                start_sample=max(0,int(start*TARGET_SR))
                end_sample=int(end*TARGET_SR)

                out_path=f"AIR/normwav/{split}/{entry['filename'].replace('.wav','')}_part{event_idx}.wav"
                os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok=True)

                # normalize audio
                segment=y[start_sample:end_sample]

                segment=segment/np.max(np.abs(segment)+1e-10)

                sf.write(out_path,segment,TARGET_SR)