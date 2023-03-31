import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='hico')
    parser.add_argument('--num_dec', type=int, default=2)

    args = parser.parse_args()

    return args


def main(args):
    ps = torch.load(args.load_path)

    obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,  # 12
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
               82, 84, 85, 86, 87, 88, 89, 90]  # 8

    for k in list(ps['model'].keys()):
        if 'decoder.layers' in k:
            ps['model'][k.replace('decoder.layers', 'decoder.sub_layers')] = ps['model'][k].clone()
            ps['model'][k.replace('decoder.layers', 'decoder.hoi_layers')] = ps['model'][k].clone()
            print(k)
        if 'decoder.norm' in k:
            ps['model'][k.replace('decoder.norm', 'decoder.sub_norm')] = ps['model'][k].clone()
            ps['model'][k.replace('decoder.norm', 'decoder.hoi_norm')] = ps['model'][k].clone()
            print(k)
        if 'decoder.ref_point_head' in k:
            ps['model'][k.replace('decoder.ref_point_head', 'decoder.sub_ref_point_head')] = ps['model'][k].clone()
            print(k)
        if 'bbox_embed' in k:
            ps['model'][k.replace('bbox_embed', 'sub_bbox_embed')] = ps['model'][k].clone()
            print(k)
        if 'tgt_embed' in k:
            ps['model'][k.replace('tgt_embed', 'sub_tgt_embed')] = ps['model'][k].clone()
            ps['model'][k.replace('tgt_embed', 'hoi_tgt_embed')] = ps['model'][k].clone()
            print(k)
        if 'enc_out_bbox_embed' in k:
            ps['model'][k.replace('enc_out_bbox_embed', 'enc_out_sub_bbox_embed')] = ps['model'][k].clone()
            print(k)

    ps['model']['transformer.enc_out_class_embed.weight'] = ps['model']['transformer.enc_out_class_embed.weight'].clone()[obj_ids]
    ps['model']['transformer.enc_out_class_embed.bias'] = ps['model']['transformer.enc_out_class_embed.bias'].clone()[obj_ids]

    for i in range(args.num_dec):
        ps['model']['class_embed.' + str(i) + '.weight'] = ps['model']['class_embed.' + str(i) + '.weight'].clone()[obj_ids]
        ps['model']['class_embed.' + str(i) + '.bias'] = ps['model']['class_embed.' + str(i) + '.bias'].clone()[obj_ids]

    torch.save(ps, args.save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
