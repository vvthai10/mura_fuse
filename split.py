import os
import shutil
from pathlib import Path

from PIL import Image


def is_valid_image(file_name):
    if not os.path.isfile(file_name):
        return False
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False


def get_all_files(dirpath):
    return sum(
        [
            [os.path.join(os_walks[0], f) for f in os_walks[2]]
            for os_walks in os.walk(dirpath)
        ],
        [],
    )


def get_all_images(dirpath):
    return [p for p in get_all_files(dirpath) if is_valid_image(p)]


def copy(src_path, dst_path):
    os.makedirs(dst_path.parent, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


all_files = get_all_images("./data/heatmap-lqn")

split1 = [
    p.replace("./data/heatmap-lqn", "heatmap-lqn-1")
    for p in all_files[: int(len(all_files) / 2)]
]

for i, p in enumerate(split1):
    os.makedirs(Path(p).parent, exist_ok=True)
    shutil.copyfile(all_files[i], p)

split2 = [
    p.replace("./data/heatmap-lqn", "heatmap-lqn-2")
    for p in all_files[int(len(all_files) / 2) :]
]

for i, p in enumerate(split2):
    os.makedirs(Path(p).parent, exist_ok=True)
    shutil.copyfile(all_files[i + len(split1)], p)
