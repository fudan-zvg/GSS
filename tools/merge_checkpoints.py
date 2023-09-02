import torch
import argparse

def merge_checkpoints(model_path, post_model_path, target_path, backbone_type):
    """
    Merge two checkpoints based on the backbone type.

    Args:
        model_path (str): Path to the segmentation model checkpoint.
        post_model_path (str): Path to the post-processing model checkpoint.
        target_path (str): Path to save the merged checkpoint.
        backbone_type (str): Type of the backbone (e.g., 'swin', 'hrnet').
    """
    model_ckp = torch.load(model_path)
    post_model_ckp = torch.load(post_model_path)

    composite_ckp = dict()
    composite_ckp['meta'] = model_ckp['meta']
    composite_ckp['optimizer'] = model_ckp['optimizer']
    composite_ckp['state_dict'] = model_ckp['state_dict']

    if backbone_type == 'swin':
        for key, value in post_model_ckp['state_dict'].items():
            if key in model_ckp['state_dict']:
                print(f"[Seg model] {key} is from seg model checkpoint")
            else:
                composite_ckp['state_dict'][key] = post_model_ckp['state_dict'][key]
                print(f"[Post model] {key} is from color2cls model checkpoint")
    else:
        for key, value in model_ckp['state_dict'].items():
            print(f"[Seg model] {key}")
        for key, value in post_model_ckp['state_dict'].items():
            if 'decode_head' in key and key not in model_ckp['state_dict'] and 'pixel' not in key:
                composite_ckp['state_dict'][key] = post_model_ckp['state_dict'][key]
                print(f"[Post model] {key}")

    torch.save(composite_ckp, target_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge two checkpoints based on the backbone type.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the segmentation model checkpoint.")
    parser.add_argument("--post_model_path", type=str, required=True, help="Path to the post-processing model checkpoint.")
    parser.add_argument("--target_path", type=str, required=True, help="Path to save the merged checkpoint.")
    parser.add_argument("--backbone_type", type=str, required=True, choices=['swin', 'hrnet'], help="Type of the backbone (e.g., 'swin', 'hrnet').")

    args = parser.parse_args()
    merge_checkpoints(args.model_path, args.post_model_path, args.target_path, args.backbone_type)

# run the following command to merge the checkpoints
# python merge_checkpoints.py --model_path /path/to/model_checkpoint --post_model_path /path/to/post_model_checkpoint --target_path /path/to/target_checkpoint --backbone_type swin
