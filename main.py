import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2, wide_resnet101_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, de_wide_resnet101_2, Classifier
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation
from torch.nn import functional as F
from torchvision import transforms as T
from mvtec import SelfSupMVTecDataset, CLASS_NAMES, TEXTURES, OBJECTS
import tqdm

WIDTH_BOUNDS_PCT = {
    'bottle': ((0.03, 0.4), (0.03, 0.4)),
    'cable': ((0.01, 0.05), (0.01, 0.1)),
    'capsule': ((0.01, 0.05), (0.01, 0.05)),
    'hazelnut': ((0.03, 0.35), (0.03, 0.35)),
    'metal_nut': ((0.01, 0.05), (0.01, 0.01)),
    'pill': ((0.01, 0.05), (0.01, 0.05)),
    'screw': ((0.01, 0.05), (0.01, 0.05)),
    'toothbrush': ((0.01, 0.01), (0.01, 0.01)),
    'transistor': ((0.03, 0.4), (0.03, 0.4)),
    'zipper': ((0.01, 0.05), (0.01, 0.01)),
    'carpet': ((0.03, 0.4), (0.03, 0.4)),
    'grid': ((0.01, 0.05), (0.01, 0.1)),
    'leather': ((0.03, 0.4), (0.03, 0.4)),
    'tile': ((0.01, 0.05), (0.01, 0.1)),
    'wood': ((0.01, 0.05), (0.01, 0.01))}

MIN_OVERLAP_PCT = {
    'bottle': 0.25,
    'capsule': 0.25,
    'hazelnut': 0.25, 'metal_nut': 0.25, 'pill': 0.25,
    'screw': 0.25, 'toothbrush': 0.25,
    'zipper': 0.25}

MIN_OBJECT_PCT = {
    'bottle': 0.7,
    'capsule': 0.7,
    'hazelnut': 0.7, 'metal_nut': 0.5, 'pill': 0.7,
    'screw': .5, 'toothbrush': 0.25,
    'zipper': 0.7}

NUM_PATCHES = {

    'bottle': 3,
    'cable': 3,
    'capsule': 1, 'hazelnut': 3, 'metal_nut': 1,
    'pill': 1, 'screw': 1, 'toothbrush': 1,
    'transistor': 3, 'zipper': 1,
    'carpet': 4, 'grid': 4,
    'leather': 4,
    'tile': 4,
    'wood': 4}

INTENSITY_LOGISTIC_PARAMS = {
    'bottle': (1 / 12, 24),
    'cable': (1 / 12, 24),
    #
    'capsule': (1 / 2, 4), 'hazelnut': (1 / 12, 24), 'metal_nut': (1 / 3, 7),
    'pill': (1 / 3, 7), 'screw': (1, 3), 'toothbrush': (1 / 6, 15),
    'transistor': (1 / 6, 15), 'zipper': (1 / 6, 15),
    'carpet': (1 / 3, 7), 'grid': (1 / 3, 7), 'leather': (1 / 3, 7), 'tile': (1 / 3, 7),
    'wood': (1 / 6, 15)}

UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

BACKGROUND = {
    'bottle': (200, 60), 'screw': (200, 60),
    'capsule': (200, 60), 'zipper': (200, 60),
    'hazelnut': (20, 20), 'pill': (20, 20), 'toothbrush': (20, 20), 'metal_nut': (20, 20)}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([fs_list[0].shape[0], 1, out_size, out_size])
    else:
        anomaly_map = np.ones([fs_list[0][0].shape[0], 1, out_size, out_size])
    a_map_list = []
    a = ft_list[0]
    for i in range(len(a)):
        fs = a[i]
        ft = a[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[:, :, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_path = './datasets/mvtec/'

    test_path = './datasets/mvtec/'


    BACKGROUND = {'bottle': (200, 60), 'screw': (200, 60), 'capsule': (200, 60), 'zipper': (200, 60),
                  'hazelnut': (20, 20), 'pill': (20, 20), 'toothbrush': (20, 20), 'metal_nut': (20, 20)}

    if _class_ in UNALIGNED_OBJECTS:
        train_transform = T.Compose([
            T.CenterCrop(256),
            T.RandomCrop(256)
        ])
        res = 256


    elif _class_ in OBJECTS:
        train_transform = T.Compose([
            T.CenterCrop(256),
            T.RandomCrop(256)])
        res = 256
    else:
        train_transform = T.Compose([
            T.CenterCrop(256),
            T.RandomCrop(256),
        ])
        res = 256

    train_dat = SelfSupMVTecDataset(root_path=train_path, class_name=_class_, is_train=True,
                                    low_res=res, download=False, transform=train_transform)

    train_dat.configure_self_sup(self_sup_args={'gamma_params': (2, 0.05, 0.03), 'resize': False,
                                                'shift': True, 'same': True, 'mode': 'swap', 'label_mode': 'binary'})
    train_dat.configure_self_sup(self_sup_args={'skip_background': BACKGROUND.get(_class_)})
    train_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(_class_),
                                                         'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(
                                                             _class_),
                                                         'num_patches': NUM_PATCHES.get(_class_),
                                                         'min_object_pct': MIN_OBJECT_PCT.get(_class_),
                                                         'min_overlap_pct': MIN_OVERLAP_PCT.get(_class_)})

    train_dataloader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=0,
                                  worker_init_fn=lambda _: np.random.seed(
                                      torch.utils.data.get_worker_info().seed % 2 ** 32))
    test_dat = SelfSupMVTecDataset(root_path=test_path, class_name=_class_, is_train=False,
                                   low_res=res, download=False, transform=train_transform)

    test_dat.configure_self_sup(self_sup_args={'gamma_params': (2, 0.05, 0.03), 'resize': False,
                                               'shift': True, 'same': True, 'mode': 'swap', 'label_mode': 'binary'})
    test_dat.configure_self_sup(self_sup_args={'skip_background': BACKGROUND.get(_class_)})

    test_dat.configure_self_sup(on=True, self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(_class_),
                                                        'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(
                                                            _class_),
                                                        'num_patches': NUM_PATCHES.get(_class_),
                                                        'min_object_pct': MIN_OBJECT_PCT.get(_class_),
                                                        'min_overlap_pct': MIN_OVERLAP_PCT.get(_class_)})

    test_dataloader = DataLoader(test_dat, batch_size=1, shuffle=False, num_workers=0,
                                 worker_init_fn=lambda _: np.random.seed(
                                     torch.utils.data.get_worker_info().seed % 2 ** 32))

    encoder, bn = wide_resnet50_2(pretrained=True)

    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(_class_, pretrained=False)
    decoder = decoder.to(device)
    # ckp_path = './checkpoints/' + 'wres50_' + _class_ + '_200' + '.pth'
    # bn.load_state_dict(torch.load(ckp_path)["bn"], strict=False)

    # decoder.load_state_dict(torch.load(ckp_path)["decoder"], strict=False)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    for epoch in range(200):
        bn.train()
        decoder.train()
        loss_list = []
        for batch_idx, (img, aug_img, _, label, src_object_mask, mask0, mask_1, mask_2, mask_3) in enumerate(
                tqdm(train_dataloader, desc='Batch') if False else train_dataloader):
            img = img.to(device)
            aug_img = aug_img.to(device)
            with torch.no_grad():
                inputs = encoder(img)
                aug_inputs = encoder(aug_img)
            outputs = decoder(bn(aug_inputs))
            loss = loss_fucntion(
                    [inputs[0], inputs[1], inputs[2]], [outputs[0], outputs[1], outputs[2]])
            optimizer.zero_grad()
            #
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if epoch + 1 == 200:
            auroc_px, auroc_sp, aupro_px, loss_, img_fpr, img_tpr, fpr, tpr = evaluation(
                encoder, bn, decoder,
                test_dataloader, device)
            auroc_px_list.append(auroc_px)

            auroc_sp_list.append(auroc_sp)

            aupro_px_list.append(aupro_px)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}, loss{:.3}'.format(auroc_px, auroc_sp,
                                                                                               aupro_px, loss_))
            auroc_px_fin = sum(auroc_px_list) / len(auroc_px_list)
            auroc_sp_fin = sum(auroc_sp_list) / len(auroc_sp_list)
            aupro_px_fin = sum(aupro_px_list) / len(aupro_px_list)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px_fin, auroc_sp_fin,
                                                                                    aupro_px_fin))
    return auroc_px_fin, auroc_sp_fin, aupro_px_fin


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='The name of the dataset to perform tests on.'
                             'Choose among `mnist`, `cifar10`, `FashionMNIST`')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    args = parser.parse_args()
    torch.cuda.empty_cache()
    setup_seed(111)

    item_list = [
        'bottle',
        'cable',
        'leather',
        'capsule',
        'grid',
        'pill',
        'carpet',
        'hazelnut',
        'transistor',
        'metal_nut',
        'screw',
        'toothbrush',
        'zipper',
        'tile',
        'wood'
    ]
    total_roc_auc = []
    total_pixel_roc_auc = []
    total_aupro_px = []
    for i in item_list:
        auroc_px, auroc_sp, aupro_px = train(i)
        total_roc_auc.append(auroc_px)
        total_pixel_roc_auc.append(auroc_sp)
        total_aupro_px.append(aupro_px)
    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))

    print('Average PRO: %.3f' % np.mean(total_aupro_px))

    values = {'Average ROCAUC': np.mean(total_roc_auc), 'Average pixel ROCUAC': np.mean(total_pixel_roc_auc),
              'Average PRO': np.mean(total_aupro_px)}


