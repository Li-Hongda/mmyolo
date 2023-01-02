import argparse
import os.path as osp
from collections import OrderedDict

import torch

neck_convert_dict = {
    'merge': ['merge_3', 'merge_4', 'merge_5', 'merge_6', 'merge_7'],
    'conv2.conv': 'conv_main.conv',
    'conv2.bn': 'conv_main.bn',
    'conv1.conv': 'conv_short.conv',
    'conv1.bn': 'conv_short.bn',
    'conv3.conv': 'conv_final.conv',
    'conv3.bn': 'conv_final.bn',
}
convert_dict_t = {
    'backbone.block_list.0': 'backbone.stem',
    'backbone.block_list.1': 'backbone.stage1.0',
    'backbone.block_list.2': 'backbone.stage2.0',
    'backbone.block_list.3': 'backbone.stage3.0',
    'backbone.block_list.4': 'backbone.stage4.0',
    'backbone.block_list.5': 'backbone.stage5.0',
    'head': 'bbox_head.head_module',
    'conv_start': 'conv_main',
    'conv_shortcut': 'conv_short',
    'conv_fuse': 'conv_final',
    'conv1.conv1': 'conv1.conv',
    'conv1.bn1': 'conv1.bn'
}
convert_dict_m = {
    'backbone.csp_stage.0': 'backbone.stem',
    'backbone.csp_stage.1': 'backbone.stage1.0',
    'backbone.csp_stage.2': 'backbone.stage2.0',
    'backbone.csp_stage.3': 'backbone.stage3.0',
    'backbone.csp_stage.4': 'backbone.stage4.0',
    'downsampler.conv.conv1': 'downsampler.conv',
    'downsampler.conv.bn1': 'downsampler.bn',
    'conv_start.conv.conv1': 'conv_main.conv',
    'conv_start.conv.bn1': 'conv_main.bn',
    'conv_shortcut.conv.conv1': 'conv_short.conv',
    'conv_shortcut.conv.bn1': 'conv_short.bn',
    'conv_fuse.conv.conv1': 'conv_final.conv',
    'conv_fuse.conv.bn1': 'conv_final.bn',
    'conv1.conv1': 'conv1.conv',
    'conv1.bn1': 'conv1.bn'
}


def convert_backbone(model_key, model_weight, state_dict, converted_names,
                     convert_dict):

    new_key = model_key
    for old, new in convert_dict.items():
        if old in model_key:
            new_key = new_key.replace(old, new)
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_neck(model_key, model_weight, state_dict, converted_names):
    new_key = model_key
    if model_key[5:12] in neck_convert_dict[
            'merge'] and 'convs' not in model_key:
        for old, new in neck_convert_dict.items():
            if old == 'merge':
                continue
            elif old in model_key:
                new_key = new_key.replace(old, new)
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_head(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('head', 'bbox_head.head_module')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


convert_dicts = {
    'damoyolo_tinynasL20_T': convert_dict_t,
    'damoyolo_tinynasL20_T_418': convert_dict_t,
    'damoyolo_tinynasL25_S': convert_dict_t,
    'damoyolo_tinynasL25_S_456': convert_dict_t,
    'damoyolo_tinynasL35_M': convert_dict_m,
    'damoyolo_tinynasL35_M_487': convert_dict_m,
}


def convert(src, dst):
    """Convert keys in detectron pretrained DAMO-YOLO models to mmyolo
    style."""
    blobs = torch.load(src)['model']
    state_dict = OrderedDict()
    converted_names = set()
    convert_dict = convert_dicts[osp.basename(src)[:-4]]
    for key, weight in blobs.items():
        if 'backbone.' in key:
            convert_backbone(key, weight, state_dict, converted_names,
                             convert_dict)
        elif 'neck.' in key:
            convert_neck(key, weight, state_dict, converted_names)
        elif 'head' in key:
            convert_head(key, weight, state_dict, converted_names)

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src',
        default='work_dirs/damo_yolo_t_coco/damoyolo_tinynasL20_T.pth',
        help='src model path')
    parser.add_argument(
        '--dst',
        default='work_dirs/damo_yolo_t_coco/damoyolo_T_coco.pth',
        help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
