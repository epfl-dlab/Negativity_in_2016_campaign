#!/usr/bin/env python3
import bz2
import json
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def main():
    """
    Extracts most relevant information from a Wikidata dump - use the latest-all.json.bzip2 dump you can get at:
    https://dumps.wikimedia.org/wikidatawiki/entities/
    """
    # Do not enforce encoding here since the input encoding is correct
    fails = 0
    with open(args.output, "w") as output_file:
        with bz2.open(args.input, 'rb') as bzinput:
            for i, line in tqdm(enumerate(bzinput)):
                line = line.decode('utf-8')
                try:
                    s = json.loads(line.strip(",\n"))
                except json.JSONDecodeError:
                    fails += 1
                    continue

                if s.get("labels", {}).get("en") is not None:
                    s["label"] = s["labels"]["en"]["value"]
                if s.get("labels") is not None:
                    del s["labels"]
                else:
                    continue

                # Occupation
                if len(s.get("claims", {}).get("P106", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P106"]:
                        # id: "Q123", "numeric-id": 123
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["occupation"] = tmp

                # Gender
                if len(s.get("claims", {}).get("P21", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P21"]:
                        # id: "Q123", "numeric-id": 123
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["gender"] = tmp

                # Country of citizenship
                if len(s.get("claims", {}).get("P27", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P27"]:
                        # id: "Q123", "numeric-id": 123
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["nationality"] = tmp

                # Position Held
                if len(s.get("claims", {}).get("P39", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P39"]:
                        # id: "Q123", "numeric-id": 123
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["positions_held"] = tmp

                # Date of Birth
                if len(s.get("claims", {}).get("P569", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P569"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("time")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["time"])
                    if len(tmp) > 0:
                        s["date_of_birth"] = tmp

                # Academic Degree
                if len(s.get("claims", {}).get("P512", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P512"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["academic_degree"] = tmp

                # Member of Political Party
                if len(s.get("claims", {}).get("P102", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P102"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["party"] = tmp

                # Candidacy in election
                if len(s.get("claims", {}).get("P3602", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P3602"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["candidacy"] = tmp

                # US Congress Bio ID
                # Get more information on the politicians based on the ID here: https://bioguide.congress.gov
                if len(s.get("claims", {}).get("P1157", [])) > 0:
                    tmp = None
                    for v in s["claims"]["P1157"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value")
                            is not None
                        ):
                            tmp = v["mainsnak"]["datavalue"]["value"]
                            break
                    if tmp is not None:
                        s["US_congress_bio_ID"] = tmp

                # Ethnic Group
                if len(s.get("claims", {}).get("P172", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P172"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["ethnic_group"] = tmp

                # Religion
                if len(s.get("claims", {}).get("P140", [])) > 0:
                    tmp = []
                    for v in s["claims"]["P140"]:
                        if (
                            v["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                            is not None
                        ):
                            tmp.append(v["mainsnak"]["datavalue"]["value"]["id"])
                    if len(tmp) > 0:
                        s["religion"] = tmp

                # Aliases. Removing leftovers and unnecessary attributes
                if len(s.get("aliases", {}).get("en", [])) > 0:
                    s["aliases"] = [v["value"] for v in s["aliases"]["en"]]
                elif s.get("aliases") is not None:
                    del s["aliases"]
                if s.get("descriptions") is not None:
                    del s["descriptions"]
                if s.get("sitelinks") is not None:
                    del s["sitelinks"]
                if s.get("claims") is not None:
                    del s["claims"]

                output_file.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Could not read {fails} lines.")


if __name__ == "__main__":
    main()
