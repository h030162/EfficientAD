#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
import cv2
from PIL import Image
import copy
import json
import pickle
def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--trainpath', default='data/train')
    parser.add_argument('-o', '--output_dir', default='output')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')    
    parser.add_argument('-e', '--epochs', type=int, default=400)
    parser.add_argument('-t', '--type', type=str, default="train")
    parser.add_argument('-i', '--input', type=str, default="test image path")
    return parser.parse_args()

seed = 2024
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256
batch_size = 1

default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

config = get_argparse()

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)    
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)
    return channel_mean, channel_std
def train():    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_train_set = ImageFolderWithoutTarget(config.trainpath,
        transform=transforms.Lambda(train_transform))
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                        [train_size,
                                                        validation_size],
                                                        rng)
    test_data_set = ImageFolderWithPath("data/test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)
    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)
    best_loss = 1e10
    best_epoch = -1
    for epoch in range(config.epochs):
        teacher.eval()
        student.train()
        autoencoder.train()
        train_total_losss = 0
        trainbar = tqdm(train_loader)
        lr = optimizer.param_groups[0]['lr']
        for image_st, image_ae in trainbar:
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()

            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st = student(image_st)[:, :out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            loss_st = loss_hard
            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            train_total_losss += loss_total.item()
            trainbar.set_description(f"epoch {epoch},current batch loss {loss_total.item():.6f}, total loss: {train_total_losss:.6f}, best loss: {best_loss:.6f}, best epoch: {best_epoch}, lr: {lr:.6f}")

        if train_total_losss < best_loss:
            best_loss = train_total_losss
            best_epoch = epoch

            teacher.eval()
            student.eval()
            autoencoder.eval()
            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(test_set=test_data_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))

            save_dict = {}
            save_dict["q_st_start"] = q_st_start.item()
            save_dict["q_st_end"] = q_st_end.item()
            save_dict["q_ae_start"] = q_ae_start.item()
            save_dict["q_ae_end"] = q_ae_end.item()
            save_dict["teacher_mean"] = teacher_mean.cpu().numpy().tolist()
            save_dict["teacher_std"] = teacher_std.cpu().numpy().tolist()
            save_dict["teacher_state_dict"] = teacher.state_dict()
            save_dict["student_state_dict"] = student.state_dict()
            save_dict["autoencoder_state_dict"] = autoencoder.state_dict()
            save_dict["best_loss"] = best_loss
            pickle.dump(save_dict, open(os.path.join(config.output_dir, 'best.pkl'), 'wb'))

            # f = open(os.path.join(config.output_dir, 'q_params.json'), 'w')
            # f.write(json.dumps(save_dict))
            # f.close()

            # print(q_st_start, q_st_end, q_ae_start, q_ae_end)
            # print(teacher_mean)
            # print(teacher_std)

            # torch.save(teacher, os.path.join(config.output_dir, 'teacher.pth'))
            # torch.save(student, os.path.join(config.output_dir, 'student_best.pth'))
            # torch.save(autoencoder, os.path.join(config.output_dir, 'autoencoder_best.pth'))

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='inference'):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        # opencv_map = copy.deepcopy(map_combined)        
        # opencv_map[opencv_map < 0] = 0
        # opencv_map[opencv_map > 1] = 1
        # opencv_map = (opencv_map * 255).astype(np.uint8)

        #print(map_combined.shape, map_combined.min(), map_combined.max(), map_combined.dtype)

        # pil_image = Image.fromarray(map_combined)
        # pil_image.save("test.png")

        #cv2.imwrite("test.png", map_combined_scaled)




        #print(map_combined.shape, map_combined.min(), map_combined.max(), map_combined.dtype)
        #cv2.convert

        #print("===debug test map_combined2:", map_combined.shape)

        # resize_map = cv2.resize(map_combined_scaled, (256, 256))
        # cv2.imshow("resize_map", resize_map)
        # cv2.waitKey(0)

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

def auc_test():
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    autoencoder = get_autoencoder(out_channels)
    weights_path = os.path.join(config.output_dir, 'best.pkl')
    if os.path.exists(weights_path):
        save_dict = pickle.load(open(weights_path, 'rb'))
        #print(save_dict.keys())
        q_st_start = save_dict["q_st_start"]
        q_st_end = save_dict["q_st_end"]
        q_ae_start = save_dict["q_ae_start"]
        q_ae_end = save_dict["q_ae_end"]
        teacher_mean = torch.tensor(save_dict["teacher_mean"])
        teacher_std = torch.tensor(save_dict["teacher_std"])
        teacher.load_state_dict(save_dict["teacher_state_dict"])
        student.load_state_dict(save_dict["student_state_dict"])
        autoencoder.load_state_dict(save_dict["autoencoder_state_dict"])
        print("load weights from:", weights_path)

        if on_gpu:
            teacher.cuda()
            student.cuda()
            autoencoder.cuda()
            teacher_mean = teacher_mean.cuda()
            teacher_std = teacher_std.cuda()
            q_st_start = torch.tensor(q_st_start).cuda()
            q_st_end = torch.tensor(q_st_end).cuda()
            q_ae_start = torch.tensor(q_ae_start).cuda()
            q_ae_end = torch.tensor(q_ae_end).cuda()
        teacher.eval()
        student.eval()
        autoencoder.eval()
        test_data_set = ImageFolderWithPath("data/test")
        auc = test(test_set=test_data_set, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start,
            q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            test_output_dir=None, desc='Inference')
        print('Image auc: {:.4f}'.format(auc))
    else:
        raise Exception("weights not found")

def single_test():
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    autoencoder = get_autoencoder(out_channels)
    weights_path = os.path.join(config.output_dir, 'best.pkl')
    if os.path.exists(weights_path):
        save_dict = pickle.load(open(weights_path, 'rb'))
        #print(save_dict.keys())
        q_st_start = save_dict["q_st_start"]
        q_st_end = save_dict["q_st_end"]
        q_ae_start = save_dict["q_ae_start"]
        q_ae_end = save_dict["q_ae_end"]
        teacher_mean = torch.tensor(save_dict["teacher_mean"])
        teacher_std = torch.tensor(save_dict["teacher_std"])
        teacher.load_state_dict(save_dict["teacher_state_dict"])
        student.load_state_dict(save_dict["student_state_dict"])
        autoencoder.load_state_dict(save_dict["autoencoder_state_dict"])
        print("load weights from:", weights_path)

        if on_gpu:
            teacher.cuda()
            student.cuda()
            autoencoder.cuda()
            teacher_mean = teacher_mean.cuda()
            teacher_std = teacher_std.cuda()
            q_st_start = torch.tensor(q_st_start).cuda()
            q_st_end = torch.tensor(q_st_end).cuda()
            q_ae_start = torch.tensor(q_ae_start).cuda()
            q_ae_end = torch.tensor(q_ae_end).cuda()
        teacher.eval()
        student.eval()
        autoencoder.eval()

        image = Image.open(config.input)
        cv_source = np.array(image)
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        #map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        basename = os.path.basename(config.input)
        cv_map = copy.deepcopy(map_combined)        
        cv_map[cv_map < 0] = 0
        cv_map[cv_map > 1] = 1
        cv_map = (cv_map * 255).astype(np.uint8)
        cv_map_threshold = cv2.threshold(cv_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite(f"map_{basename}", cv_map_threshold)
        #cv2.imshow("map_combined", map_combined)
        cv_source = cv2.resize(cv_source, (map_combined.shape[1], map_combined.shape[0]))
        #cv2.imshow("source", cv_source)
        cv_combine = copy.deepcopy(cv_source)
        cv_combine[:,:,0] = cv_combine[:,:,0] + cv_map_threshold
        cv_combine[:,:,0][cv_map_threshold == 255]=255
        cv2.imwrite(f"combine_{basename}", cv_combine)
        #cv2.waitKey(0)



if __name__ == '__main__':
    if config.type == 'train':
        train()
    elif config.type == 'auc':
        auc_test()
    elif config.type == 'test':
        single_test()
    else:
        pass