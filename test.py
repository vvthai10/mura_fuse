import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import resnet50, fusenet
import shutil
from pathlib import Path

Trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

Normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])


def generate_grad_cam(net, ori_image):
    """
    :param net: deep learning network(ResNet DataParallel object)
    :param ori_image: the original image
    :return: gradient class activation map
    """
    input_image = Trans(ori_image)

    feature = None
    gradient = None

    def func_f(module, input, output):
        nonlocal feature
        feature = output.data.cpu().numpy()

    def func_b(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].data.cpu().numpy()

    # print(net.module)
    net.module.global_branch.layer4.register_forward_hook(func_f)
    net.module.global_branch.layer4.register_backward_hook(func_b)

    out = net(input_image.unsqueeze(0))

    pred = (out.data > 0.5)

    net.zero_grad()

    loss = F.binary_cross_entropy(out, pred.float())
    loss.backward()

    feature = np.squeeze(feature, axis=0)
    gradient = np.squeeze(gradient, axis=0)

    weights = np.mean(gradient, axis=(1, 2), keepdims=True)

    cam = np.sum(weights * feature, axis=0)

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = 1.0 - cam
    cam = np.uint8(cam * 255)

    return cam


def localize(cam_feature, ori_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by generate_grad_cam
    :param ori_image: the original image
    :return: img with heatmap, the abnormality region is highlighted
    """
    ori_image = np.array(ori_image)
    activation_heatmap = cv2.applyColorMap(cam_feature, cv2.COLORMAP_JET)
    activation_heatmap = cv2.resize(activation_heatmap, (ori_image.shape[1], ori_image.shape[0]))
    img_with_heatmap = 0.15 * np.float32(activation_heatmap) + 0.85 * np.float32(ori_image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap) * 255
    return img_with_heatmap


def localize2(cam_feature, ori_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by generate_grad_cam
    :param ori_image: input of the network
    :return: img with heatmap, the abnormality region is in a red window
    """
    ori_image = np.array(ori_image)
    cam_feature = cv2.resize(cam_feature, (ori_image.shape[1], ori_image.shape[0]))
    crop = np.uint8(cam_feature > 0.7 * 255)
    h = ori_image.shape[0]
    w = ori_image.shape[1]
    ret, markers = cv2.connectedComponents(crop)
    branch_size = np.zeros(ret)
    for i in range(h):
        for j in range(w):
            t = int(markers[i][j])
            branch_size[t] += 1
    branch_size[0] = 0
    max_branch = np.argmax(branch_size)
    mini = h
    minj = w
    maxi = -1
    maxj = -1
    for i in range(h):
        for j in range(w):
            if markers[i][j] == max_branch:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    img_with_window = np.uint8(ori_image)
    img_with_window[mini:mini+2, minj:maxj, 0:1] = 255
    img_with_window[mini:mini+2, minj:maxj, 1:3] = 0
    img_with_window[maxi-2:maxi, minj:maxj, 0:1] = 255
    img_with_window[maxi-2:maxi, minj:maxj, 1:3] = 0
    img_with_window[mini:maxi, minj:minj+2, 0:1] = 255
    img_with_window[mini:maxi, minj:minj+2, 1:3] = 0
    img_with_window[mini:maxi, maxj-2:maxj, 0:1] = 255
    img_with_window[mini:maxi, maxj-2:maxj, 1:3] = 0

    return img_with_window


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


def copy(src_path, dst_path):
    os.makedirs(dst_path.parent, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def get_all_images(dirpath):
    return [p for p in get_all_files(dirpath) if is_valid_image(p)]


if __name__ == '__main__':
    # net = resnet50(pretrained=True)

    # net.load_state_dict(
    #     torch.load("./output/lqn_mura_v2/model/fuse_start.pth.tar")
    # )
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load("./models/fuse_start.pth.tar")
    # print(checkpoint['state_dict'].keys())
    
    global_branch = torch.load("./output/lqn_mura_v2/model/best_model.pth.tar")['net']
    local_branch = torch.load("./output/lqn_mura_v2/model/best_model.pth.tar")['net']
    net = fusenet(global_branch, local_branch)
    net = torch.nn.DataParallel(net)

    imgs = get_all_images("./data/processed-lqn")[:1]
    for img_path in tqdm(imgs, desc="Localize "):
        ori_image = Image.open(img_path).convert('RGB')
        cam_feature = generate_grad_cam(net, ori_image)
        result1 = localize(cam_feature, ori_image)
        # result2 = localize2(cam_feature, ori_image)
        # result2 = Image.fromarray(result2)

        new_path = img_path.replace("processed-lqn", "test")
        os.makedirs(Path(new_path).parent, exist_ok=True)

        cv2.imwrite(new_path, result1)
        print(f"Done {new_path}")
        # result2.save(img_path[:-4] + "_w.png")
