#!/usr/bin/env python3

import glob
import json
import os
from datetime import datetime
from collections import defaultdict
import argparse
import re
import xml.etree.ElementTree as et
from bids_wrapper import exit_program_early

def get_locals_from_xml(xml_path: str) -> set:
    """
    Read in the xml file to find the localizers.

    :param xml_path: Path to XML generated by XNAT
    :type xml_path: str
    :return: Set containing all localizer IDs
    :rtype: set
    """
    tree = et.parse(xml_path)
    prefix = "{" + str(tree.getroot()).split("{")[-1].split("}")[0] + "}"
    scans = tree.getroot().find(
        f"./{prefix}experiments/{prefix}experiment/{prefix}scans"
    )
    print(scans)
    localizers = set()
    for s in scans:
        if re.match(r"Localizer.*", s.get("type")) and s.find(f'{prefix}quality').text == "usable":
            localizers.add(int(s.get("ID")))
    return sorted(localizers)


def get_func_from_bids(bids_path: str,
                       localizers: set[int],
                       json_dict: defaultdict[list],
                       groupings: list[dict[str:set]]):
    """
    Read in the JSON associated with task-based runs from the bids dir, and pair with a localizer run.

    :param bids_path: path to BIDS-compliant .nii/.json files
    :type bids_path: str
    :param localizers: set of localizer IDs
    :type localizers: set[int]
    :param json_dict: dict with each key being a series number, each value being an object representation of it's JSON file with added 'bidsname' and 'filename' attributes
    :type json_dict: defaultdict[list]
    :param groupings: list of dictionaries containing mappings of localizers to series number
    :type groupings: list[dict[str:set]]
    """
    bids_func_json = sorted(glob.glob(bids_path+"/func/*bold.json"))
    # assert len(bids_func_json) > 0, "---[ERROR]: Could not find any JSON files for bold runs in the functional directory of the bids data"
    if len(bids_func_json) == 0:
        exit_program_early("Could not find any JSON files for bold runs in the functional directory of the bids data.")
    for jf in bids_func_json:
        jd = None
        with open(jf, "r") as j:
            jd = json.load(j)
        series_num = int(jd["SeriesNumber"])
        jd["bidsname"] = f"ses{bids_path.split('ses')[-1]}/func/{jf.split('/')[-1][:-4]}nii.gz"
        jd["filename"] = jf
        json_dict[series_num].append(jd)
        for i, l in enumerate(localizers):
            if i < len(localizers)-1:
                if series_num > l and series_num < localizers[i+1]:
                    groupings[i]["task"].add(series_num)
                    break
            else:
                groupings[i]["task"].add(series_num)


# Read in the json for the field maps from the bids dir
def get_fmap_from_bids(bids_path: str,
                       localizers: set[int],
                       json_dict: defaultdict[list],
                       groupings: list[dict[str:set]]):
    """
    Read in all fmap JSON from the bids dir, and pair to a localizer.

    :param bids_path: path to BIDS-compliant .nii/.json files
    :type bids_path: str
    :param localizers: set of localizer IDs
    :type localizers: set[int]
    :param json_dict: dict with each key being a series number, each value being an object representation of it's JSON file with added 'bidsname' and 'filename' attributes
    :type json_dict: defaultdict[list]
    :param groupings: list of dictionaries containing mappings of localizers to series number
    :type groupings: list[dict[str:set]]
    """
    bids_fmap_json = sorted(glob.glob(bids_path+"/fmap/*.json"))
    # assert len(bids_fmap_json) > 0, "---[ERROR]: Could not find any JSON files from the fieldmap directory of the bids data"
    if len(bids_fmap_json) == 0:
        exit_program_early("Could not find any JSON files from the fieldmap directory of the bids data.")
    for jf in bids_fmap_json:
        jd = None
        with open(jf, "r") as j:
            jd = json.load(j)
        series_num = int(jd["SeriesNumber"])
        jd["filename"] = jf.split("/")[-1]
        json_dict[series_num].append(jd)
        direction = "fmapAP" if jd["PhaseEncodingDirection"] == "j-" else "fmapPA"
        for i, l in enumerate(localizers):
            if i < len(localizers)-1:
                if series_num > l and series_num < localizers[i+1]:
                    groupings[i][direction].add(series_num)
                    break
            else:
                groupings[i][direction].add(series_num)


def map_fmap_to_func(xml_path: str,
                     bids_dir_path: str):
    """
    Group field maps to BOLD task runs.

    :param xml_path: path to XML generated by XNAT
    :type xml_path: str
    :param bids_dir_path: path to BIDS-compliant session directory
    :type bids_dir_path: str
    """
    print("####### Pairing field maps to functional runs #######")

    if not os.path.isfile(xml_path):
        exit_program_early(f"Session xml file {xml_path} does not exist.")
    if not os.path.isdir(bids_dir_path):
        exit_program_early(f"Session bids dicrectory {bids_dir_path} does not exist.")
    series_json = defaultdict(list)

    locals_series = get_locals_from_xml(xml_path)
    print(f"Localizers: {locals_series}")

    groups = [{"task":set(), "fmapAP": set(), "fmapPA": set()} for _ in locals_series]
    get_func_from_bids(bids_dir_path, locals_series, series_json, groups)
    get_fmap_from_bids(bids_dir_path, locals_series, series_json, groups)
    print(f"Localizer groups: {groups}") 

    for group in groups:
        # assert len(group["fmapAP"]) == len(group["fmapPA"]), "Unequal number of AP and PA field maps!"
        if len(group["fmapAP"]) != len(group["fmapPA"]):
            exit_program_early("Unequal number of AP and PA field maps.")
        fmap_pairs = tuple(zip(sorted(group["fmapAP"]), sorted(group["fmapPA"])))
        fmap_times = []

        # get times for field maps
        for i,p in enumerate(fmap_pairs):
            times = sorted((datetime.strptime(series_json[p[0]][0]["AcquisitionTime"], "%H:%M:%S.%f"),
                            datetime.strptime(series_json[p[1]][0]["AcquisitionTime"], "%H:%M:%S.%f")))
            fmap_times.append(times[0] + (abs(times[1] - times[0])/2))
        
        # pair task runs with field maps based on closest Acquisition Time
        map_pairings = {s:[] for s in fmap_pairs}
        for t in group["task"]:
            aqt = datetime.strptime(series_json[t][0]["AcquisitionTime"], "%H:%M:%S.%f")
            diff = list(map(lambda x: abs(x-aqt), fmap_times))
            pairing = fmap_pairs[diff.index(min(diff))]
            map_pairings[pairing].append(t)
        print(f"Field map pairings: {map_pairings}")

        # add the list of task run files that are paried with each field map in their json files
        for k,v in map_pairings.items():
            for s in k:
                jd = series_json[s][0]
                file = jd.pop("filename")
                jd["IntendedFor"] = []
                for t in v:
                    for echo in series_json[t]:
                        jd["IntendedFor"].append(echo["bidsname"])
                with open(f"{bids_dir_path}/fmap/{file}", "w") as out_file:
                    out_file.write(json.dumps(jd, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="group_series.py", 
        description="Grouping field maps to BOLD task runs"
    )
    parser.add_argument("xml_ses_file", help="The path to the xml file for this session")
    parser.add_argument("bids_ses_dir", help="The path to the bids directory for this session")
    args = parser.parse_args()

    xml_path = args.xml_ses_file
    bidsdir_path = args.bids_ses_dir
    
    if bidsdir_path.endswith("/"):
        bidsdir_path = bidsdir_path[:-1]

    print(xml_path)
    print(bidsdir_path)

    map_fmap_to_func(xml_path, bidsdir_path)
