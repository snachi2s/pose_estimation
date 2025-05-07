#!/usr/bin/env python3
"""
Convert COCO instance segmentation format to old format with fields:
['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']

This script preserves all other data in the JSON files.
"""

import json
import os
import argparse
from collections import defaultdict

def convert_coco_to_old_format(input_json_path, output_json_path):
    """
    Convert a COCO-style dataset to the old format with fields:
    ['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']
    
    Args:
        input_json_path: Path to the input COCO JSON file
        output_json_path: Path to save the output JSON file
    """
    print(f"Converting {input_json_path} to {output_json_path}")
    
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)
    
    #dictionary to map image IDs to file names
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
    
    #groupannotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        annotations_by_image[ann['image_id']].append(ann)
    
    old_format_data = {}
    
    for key, value in coco_data.items():
        old_format_data[key] = value
    
    old_format_data['data_fields'] = ['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']
    
    data_list = []
    
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_id_to_filename:
            continue
        
        record = {
            'image': image_id_to_filename[image_id],
            'gt_bbox': [],
            'gt_class': [],
            'gt_poly': [],
            'is_crowd': []
        }
        
        for ann in annotations:
            # bbox from [x, y, width, height] to [xmin, ymin, xmax, ymax]
            if 'bbox' in ann:
                x, y, width, height = ann['bbox']
                bbox = [float(x), float(y), float(x + width), float(y + height)]
                record['gt_bbox'].append(bbox)
            else:
                record['gt_bbox'].append([])
            
            record['gt_class'].append(ann.get('category_id', 0)) #get category ID
            
            if 'segmentation' in ann:
                record['gt_poly'].append(ann['segmentation'])
            else:
                record['gt_poly'].append([])
            
            record['is_crowd'].append(1 if ann.get('iscrowd', 0) == 1 else 0)
        
        if record['gt_bbox']:
            data_list.append(record)
    
    old_format_data['data'] = data_list
    
    with open(output_json_path, 'w') as f:
        json.dump(old_format_data, f, indent=2)
    
    print(f"Conversion complete. Saved {len(data_list)} image records to {output_json_path}")
    return old_format_data

def convert_all_files(input_dir, output_dir):
    """
    Convert all JSON files in the input directory to the old format
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save output JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, f"converted_{json_file}")
        convert_coco_to_old_format(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO format to old format')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file or directory')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        convert_all_files(args.input, args.output)
    else:
        convert_coco_to_old_format(args.input, args.output)