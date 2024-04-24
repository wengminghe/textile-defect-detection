import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2 as cv
import torchvision.transforms as T
from pathlib import Path

from models.model import Model0, Model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_paths, labels = load_dataset_folder(args)

    model = Model0(**vars(args)).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()

    f_good = open(os.path.join(args.output_dir, "good.txt"), "w")
    f_anomaly = open(os.path.join(args.output_dir, "anomaly.txt"), "w")
    preds = []
    for img_path, label in tqdm(zip(img_paths, labels), total=len(img_paths)):
        label = 0 if label == 'good' else 1
        ori_image = cv.imread(img_path)

        image = preprocess(ori_image)
        image = image.to(device)
        with torch.no_grad():
            pred = model.predict(image)
        pred = pred.cpu().numpy()
        preds.append(pred)
        if pred != label:
            if label == 0:
                f_good.write(img_path)
            else:
                f_anomaly.write(img_path)
    f_good.close()
    f_anomaly.close()

    acc, pre, rec = cal_metric(preds, labels)
    print(f'Accuracy: {acc*100:.2f}%, Precision: {pre*100:.2f}%, Recall: {rec*100:.2f}%')


def preprocess(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def load_dataset_folder(args):
    img_dir = os.path.join(args.data_dir, args.class_name, 'test')

    img_types = os.listdir(img_dir)
    img_paths = []
    labels = []
    for img_type in img_types:
        img_names = os.listdir(os.path.join(img_dir, img_type))
        for img_name in img_names:
            img_paths.append(os.path.join(img_dir, img_type, img_name))
            labels.append(img_type)

    return img_paths, labels


def cal_metric(preds, labels):
    tp, fn, tn, fp = 0, 0, 0, 0
    for pred, label in zip(preds, labels):
        label = 0 if label == 'good' else 1
        if label == 1 and pred == 1:
            tp += 1
        if label == 0 and pred == 0:
            tn += 1
        if label == 1 and pred == 0:
            fn += 1
        if label == 0 and pred == 1:
            fp += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def save_images(img, mask, img_type, pred, score_map, output_dir, img_name):
    if score_map.ndim == 3:
        score_map = score_map[0]
    score_map = np.uint8(255 * score_map)
    score_map = cv.applyColorMap(score_map, cv.COLORMAP_JET)
    img = cv.resize(img, (score_map.shape[1], score_map.shape[0]), interpolation=cv.INTER_LANCZOS4)
    mask = cv.resize(mask, (score_map.shape[1], score_map.shape[0]), interpolation=cv.INTER_NEAREST)
    img = np.concatenate([img, mask, score_map], axis=1)
    label = 0 if img_type == 'good' else 1
    if label == pred:
        cv.imwrite(os.path.join(output_dir, img_type, 'true', img_name), img)
    else:
        cv.imwrite(os.path.join(output_dir, img_type, 'false', img_name), img)


def get_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--class_name', type=str, default='lace')

    # model
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--extractor', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d', 'wide_resnet50_2'])
    parser.add_argument('--pool_type', type=str, default='avg', help='pool type for extracted feature maps')
    parser.add_argument('--parallel_blocks', type=int, default=[2, 5, 8])
    parser.add_argument('--c_conds', type=int, default=[64, 64, 64])
    parser.add_argument('--clamp_alpha', type=float, default=1.9)

    # eval
    parser.add_argument('--top_k', type=float, default=0.03)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output_dir', type=str, help='output directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

