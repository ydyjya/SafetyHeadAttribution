import json
from collections import defaultdict
from collections import OrderedDict

def sort_ships_dict(dict_res, print_flag=True, sorted_item=5):
    sorted_ships = sorted(dict_res.items(), key=lambda x: x[1], reverse=True)[:sorted_item]
    if print_flag:
        for key, value in sorted_ships:
            print(f"{key}: {value}")
    ships = {}
    for k, v in sorted_ships:
        ships[f"{k[0]}-{k[1]}"] = v.item()
    return ships


def ships_res_stats(ships_res_path):
    appearance_stats = defaultdict(int)
    kl_stats = defaultdict(float)
    with open(ships_res_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for k, v in data.items():
                appearance_stats[k] += 1
                kl_stats[k] += v
    sorted_appearance_stats = sorted(appearance_stats.items(), key=lambda x: x[1], reverse=True)
    sorted_kl_stats = sorted(kl_stats.items(), key=lambda x: x[1], reverse=True)
    sorted_appearance_stats = OrderedDict(sorted_appearance_stats)
    sorted_kl_stats = OrderedDict(sorted_kl_stats)

    return sorted_appearance_stats, sorted_kl_stats
