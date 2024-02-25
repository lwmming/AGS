import argparse
import logging

import torchvision
import torchvision.transforms as T
from torch.backends import cudnn
import os
import numpy as np
from surrogate import *
from our_dataset import OUR_dataset
from utils import *
import pretrainedmodels

import torchattacks
import timm
from timm.data import resolve_data_config

'''
normally-trained CNNs and ViTs
'''
parser = argparse.ArgumentParser(description='Attack')
parser.add_argument('--n_imgs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_dir', type=str, default='./checkpoints/coco_ags/models', help="Path from where to load trained substitute model dir")
parser.add_argument('--model_pth', type=str, default='ags_100.pth', help="Path from where to load trained substitute model file")
parser.add_argument('--data_dir', type=str, default='/home/wangruikui_2020/datasets/imagenet_val', help="Path to ImageNet-validation dir")
parser.add_argument('--eps_test', type=float, default=25.5)
parser.add_argument('--gpu', type=str, default='9', help='gpu-id')

def diet_tiny():
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    return model
def diet_small():
    model = timm.create_model("deit_small_patch16_224", pretrained=True)
    return model

def vit_tiny():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    return model

def vit_small():
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    return model

class Normalize_trans(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_trans, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if torch.max(input) > 1:
            input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean.to(device=input.device)) / std.to(
            device=input.device)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    SEED = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    print(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, "result_"+str(args.model_pth)+".log" )),
            logging.StreamHandler(),
        ],
    )
    logging.info(args)

    n_imgs = args.n_imgs // 2
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = args.model_dir
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    trans = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    dataset = OUR_dataset(data_dir=args.data_dir,
                          data_csv_dir='data/selected_data.csv',
                          mode='attack',
                          img_num=n_imgs,
                          transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = Basic_SSL_Model(128).cuda()
    model.load_state_dict(torch.load('{}/{}'.format(model_dir, args.model_pth)))
    CrossParadigmAttack = torchattacks.CpAttack(model, pretrained=False, eps=args.eps_test/255, alpha=1/255, steps=300, momentum=1.0)

    # compare with pretrained ResNet50 model.
    # model_res50 = nn.Sequential(Normalize(), torchvision.models.resnet50(pretrained=True)).cuda().eval()
    # CrossParadigmAttack = torchattacks.CpAttack(model_res50, pretrained=True, eps=args.eps_test/255, alpha=1/255, steps=300, momentum=1.0)

    models = {"Resnet-152": torchvision.models.resnet152, "VGG-19": torchvision.models.vgg19_bn, "Inception-V3": torchvision.models.inception_v3,
            "DenseNet-161": torchvision.models.densenet161, "DenseNet-121": torchvision.models.densenet121,
            "WRN-101": torchvision.models.wide_resnet101_2, "MobileNet-v2": torchvision.models.mobilenet_v2,
            "senet": pretrainedmodels.__dict__['senet154']}
    transformers =  {"diet_tiny":diet_tiny, "diet_small":diet_small, "vit_tiny":vit_tiny,"vit_small":vit_small}

    t_models = {}
    for name, obj in models.items():
        if name == "senet":
            t_model = obj(num_classes=1000, pretrained='imagenet')
        else:
            t_model = obj(pretrained=True)
        t_model = nn.Sequential(Normalize(), t_model)
        t_model.to(device)
        t_model.eval()
        t_models[name] = t_model

    t_formers = {}
    for name, obj in transformers.items():
        t_former = obj()
        config = resolve_data_config({}, model=t_former)
        norm_layer = Normalize_trans(mean=config['mean'], std=config['std'])
        t_former = nn.Sequential(norm_layer, t_former)
        t_former.to(device)
        t_former.eval()
        t_formers[name] = t_former


    accs = {x : 0.0 for x in t_models}
    accs_trans = {x : 0.0 for x in t_formers}

    total = 0
    p_cnt = 0
    for data_ind, (ori_img, labels) in enumerate(dataloader):
        print(data_ind)
        ori_img = ori_img.to(device)
        labels = labels.to(device)
        total += labels.size(0)
        att_img = CrossParadigmAttack(ori_img, labels)

        for name, t_model in t_models.items():
            with torch.no_grad():
                accs[name] += torch.sum(torch.argmax(t_model(att_img), dim=1) == labels).item()

        for name, t_model in t_formers.items():
            with torch.no_grad():
                accs_trans[name] += torch.sum(torch.argmax(t_model(att_img), dim=1) == labels).item()

        if data_ind % 10 == 9:
            for name in accs:
                print('The acc of %s is %.2f' % (name, 100 * accs[name] / total))
            for name in accs_trans:
                print('The acc of %s is %.2f' % (name, 100 * accs_trans[name] / total))
            p_cnt += 1

    print('-'*30+'final'+'-'*30)
    logging.info('-'*30+'final'+'-'*30)

    for name in accs:
        print('The acc of %s is %.2f' % (name, 100 * accs[name] / total))
        logging.info('The acc of %s is %.2f' % (name, 100 * accs[name] / total))
    for name in accs_trans:
        print('The acc of %s is %.2f' % (name, 100 * accs_trans[name] / total))
        logging.info('The acc of %s is %.2f' % (name, 100 * accs_trans[name] / total))
