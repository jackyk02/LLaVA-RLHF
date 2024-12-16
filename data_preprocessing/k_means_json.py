import pickle
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
from action_processing import ActionTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stablelm-2-1_6b",
    model_max_length=2048,
    padding_side="left",
    truncation_side="right",
    use_fast=False,
)

print(tokenizer.decode([32,   6369,   1990,    264,   1217,    323,    459,  15592,  11376,
                        44658,  18328,    430,  67349,    279,   9293,   5178,    315,    264,
                        12585,  34786,   1887,     13,  14194,     25,    -220,    198,
                        60556,    279,   1510,  22695,    505,    279,  12585,    596,  33271,
                        78830,   6382,     13,    578,  12585,  34786,   6916,    374,  19969,
                        311,   2231,  14071,  22145,    389,   1948,    315,   6307,  28392,
                        13,   3639,   1957,   1288,    279,  12585,   1935,    311,  13750,
                        22829,    279,   3465,     30,  36660,   3931,   2891,     25,    578,
                        12585,   1288,   1935,    279,   1957,     25, 100035, 100022, 100017,
                        100034, 100053, 100033,  99790,    220, 100257,  14194,     25,   5321,
                        15806,    279,   4367,    315,    279,  12585,   1957,     13,    362,
                        1695,  12585,   1957,   1288,   2980,   2204,   9547,     11,   5423,
                        22639,    449,  14932,   6302,    323,   3823,  19882,    627,   5045,
                        3931,   2891,     25,  20817,    389,   1268,  12966,   1053,   2585,
                        279,  12585,   6916,    323,    279,  17985,    315,    279,   6671,
                        11,    279,   4367,   5573,    315,    279,  12585,   1957,    374]))


# Load the Excel data
file_path = 'k_means_data_octo.xlsx'
df = pd.read_excel(file_path)

json_data = []
action_tokenizer = ActionTokenizer(tokenizer)


def tokenize_from_str(action):
    action = action.replace('\n', '')
    # Split by whitespace and filter out empty strings
    action_values = [x for x in action.strip(
        '[]').replace(',', ' ').split() if x]
    action_arr = np.array([float(x) for x in action_values])
    # print(action_arr)
    return action_tokenizer(action_arr)


with open("instruction_dict.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

cnt = 0
for _, row in df.iterrows():
    if row['action0'] == 0:
        continue
    instruction = instruction_dict[int(row['index'])]
    act0 = str(tokenize_from_str(str(row['action0'])))
    act1 = str(tokenize_from_str(str(row['action1'])))
    json_object = {
        "id": int(row['index']) * 55 + int(row['pair_index']),
        "image": f"000000{int(row['index'])}.jpg",
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n shows the current observation from the robot's wrist-mounted camera. The robot manipulation arm is attempting to {instruction.lower().rstrip('.')}. What action should the robot take to effectively accomplish the task? "
            },
            {
                "from": "gpt",
                "value": "The robot should take the action: " + act0
            }
        ],
        "output_1": {
            "from": "llava",
            "value": "The robot should take the action: " + act0
        },
        "output_2": {
            "from": "llava",
            "value": "The robot should take the action: " + act1
        },
        "preference": row['winner'],  # should always select output 1/action0
        "hallucination": False,
        "flip": False,
        "length_bias": False
    }

    json_data.append(json_object)
    cnt += 1

    if cnt % 1000 == 0:
        print(f"Processed {cnt} items...")

output_file_path = 'k_means_data_octo.json'
with open(output_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Data has been successfully written to {output_file_path}")
