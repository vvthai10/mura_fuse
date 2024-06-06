# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# event_acc = EventAccumulator("./output/lqn_mura_v2/events.out.tfevents.1716556937.phu")
# event_acc.Reload()
# # Show all tags in the log file
# print(event_acc.Tags())

# # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
# w_times, step_nums, vals = zip(*event_acc.Scalars("acc"))

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

path = "output/lqn_mura_v2"
listOutput = os.listdir(path)

listDF = []
key = "loss" # tag

for tb_output_folder in listOutput:
    print(tb_output_folder)
    folder_path = os.path.join(path, tb_output_folder)
    file = os.listdir(folder_path)[0]

    tensors = []
    steps = []
    for e in tf.compat.v1.train.summary_iterator(os.path.join(folder_path, file)):
        for v in e.summary.value:
            if v.tag == key:
                tensors.append(v.tensor)
                steps.append(e.step)

    values = [tf.make_ndarray(t) for t in tensors]

    plt.plot(steps, values)

    df = pd.DataFrame(data=values)
    df.to_csv("{}.csv".format(tb_output_folder))

plt.show()