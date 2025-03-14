import json

# Load the dictionaries from JSON files
with open('coco_data_part1.json', 'r') as f:
    coco_data_part1 = json.load(f)

with open('coco_data_part2.json', 'r') as f:
    coco_data_part2 = json.load(f)

# Merge the dictionaries
coco_data_merged = {**coco_data_part1, **coco_data_part2}

# Save the merged dictionary to a new JSON file
with open('trace_capt_coco_test.json', 'w') as f:
    json.dump(coco_data_merged, f)