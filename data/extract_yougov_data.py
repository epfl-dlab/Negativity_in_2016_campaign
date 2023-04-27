#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 31.03.23
import csv
from pathlib import Path
import re

import bs4


datafile = Path(__file__).parent.joinpath('YouGov_NewsWebsites_2023_03_31.html')
exclude_re = re.compile(r'([0-9]+%)|(^$)')
website_re = re.compile(r'^(https?://)?(www\.)?[a-zA-Z-]+\.[a-zA-Z]{2,3}(/.*)?$')
writefile = Path(__file__).parent.joinpath('yougov_outlets.csv')


def main():
    website = bs4.BeautifulSoup(datafile.read_text(), 'html.parser')
    outlet_info = list()
    for line in website.find_all('a', attrs={'class': 'rankings-entities-list-item focus-state ng-star-inserted'}):
        for span in line.find_all('span'):
            text = span.text.strip()
            if exclude_re.match(text):
                continue
            outlet_info.append(text)
    outlets = {int(rank): outlet for rank, outlet in zip(outlet_info[::2], outlet_info[1::2])}
    assert len(outlets) == 65
    print("Outlets extracted. Websites must be added manually")

    with writefile.open('w') as dump:
        writer = csv.writer(dump)
        writer.writerow(['Rank', 'Outlet', 'Website'])
        for rank, outlet in outlets.items():
            if website_re.match(outlet):
                address = outlet.lower()
            else:
                address = ''
            writer.writerow([rank, outlet, address])


if __name__ == '__main__':
    main()
