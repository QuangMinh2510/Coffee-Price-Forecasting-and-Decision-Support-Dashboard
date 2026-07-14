"""
Tai nguyen file goc ONI tu NOAA CPC, KHONG xu ly gi.
Nguon: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""

import urllib.request

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
OUT_FILE = "oni.txt"   # ten file luu, doi tuy y

req = urllib.request.Request(ONI_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = resp.read()          # tai raw bytes, giu nguyen ban goc

with open(OUT_FILE, "wb") as f:
    f.write(data)

print(f"[OK] Da tai xong -> {OUT_FILE}")