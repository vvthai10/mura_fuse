import os
import pandas as pd

frame = pd.read_csv('./data/mura/valid_image_paths.csv', header=None)

frame['label'] = frame[0].apply(lambda x: 0 if 'positive' not in x else 1)

# Lưu lại DataFrame với cột mới
frame.to_csv('./data/mura/valid.csv', index=False)
