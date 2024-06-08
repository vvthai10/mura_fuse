import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import resnet50, fusenet
from utils import get_all_images
from pathlib import Path

Trans = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

Normal = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


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

    pred = out.data > 0.5

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
    activation_heatmap = cv2.resize(
        activation_heatmap, (ori_image.shape[1], ori_image.shape[0])
    )
    img_with_heatmap = 0.15 * np.float32(activation_heatmap) + 0.85 * np.float32(
        ori_image
    )
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap) * 255
    return img_with_heatmap


def heatmap2segment(cam_feature, ori_image):
    ori_image = np.array(ori_image)
    cam_feature = cv2.resize(cam_feature, (ori_image.shape[1], ori_image.shape[0]))

    crop = np.uint8(cam_feature < 0.1 * 255)

    (totalLabels, label_ids, values, centroid) = (
        cv2.connectedComponentsWithStatsWithAlgorithm(crop, 4, cv2.CV_32S, ccltype=1)
    )
    print(
        f"totalLabels: {totalLabels}, label_ids: {label_ids}, values: {values}, centroid: {centroid}"
    )

    output = np.zeros(ori_image.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv2.bitwise_or(output, componentMask)
    output = Image.fromarray(output).convert("RGB")

    return output


if __name__ == "__main__":
    # net = resnet50(pretrained=True)

    # net.load_state_dict(
    #     torch.load("./output/lqn_mura_v2/model/fuse_start.pth.tar")
    # )
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load("./models/fuse_start.pth.tar")
    # print(checkpoint['state_dict'].keys())

    
    state_dict = torch.load("./models/fuse_best_model.pth (1).tar")["state_dict"]
    net = fusenet()
    net.load_state_dict(state_dict)
    net.set_fcweights()
    net = torch.nn.DataParallel(net)

    imgs = get_all_images("./data/processed-lqn")
    for img_path in tqdm(imgs, desc="Localize "):
        ori_image = Image.open(img_path).convert("RGB")
        cam_feature = generate_grad_cam(net, ori_image)
        result1 = localize(cam_feature, ori_image)
        # result2 = localize2(cam_feature, ori_image)
        # result2 = Image.fromarray(result2)

        new_path = img_path.replace("processed-lqn", "test-fuse-0")
        os.makedirs(Path(new_path).parent, exist_ok=True)

        cv2.imwrite(new_path, result1)
        print(f"Done {new_path}")
        # result2.save(img_path[:-4] + "_w.png")
