"""Pack NYC street centerlines (NYC Open Data inkn-q76z, CSV export) into roads.bin.

Usage: python process_roads.py roads.csv OUT_DIR
Record: <type u8, width u8, n u16> then n int16 (x, y) metre pairs. Types: 1 street, 2 highway, 3 bridge, 4 ramp.
Used to bake the road layer of ground.webp and to place bridge decks.
"""
import csv, struct, os, sys, numpy as np, shapely
csv.field_size_limit(10**9)
LON0, LAT0 = -73.975, 40.703; COS = np.cos(np.radians(LAT0)); MPD = 111320.0
def proj(c):
    c = c.copy(); c[:,0] = (c[:,0]-LON0)*COS*MPD; c[:,1] = (c[:,1]-LAT0)*MPD; return c
def num(s):
    try: return float(s.replace(',',''))
    except: return 0.0
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
r = csv.reader(open(src, newline='', encoding='utf-8')); h = next(r); ix = {k:i for i,k in enumerate(h)}
keep = {'1':1,'2':2,'3':3,'9':4,'13':1,'10':1}
rows = list(r)
geoms = shapely.transform(shapely.from_wkt(np.array([x[ix['the_geom']] for x in rows], dtype=object), on_invalid='ignore'), proj)
with open(os.path.join(out, 'roads.bin'), 'wb') as w:
    for row, g in zip(rows, geoms):
        t = row[ix['RW_TYPE']].strip()
        if t not in keep or g is None: continue
        for part in shapely.get_parts(g):
            co = np.round(shapely.get_coordinates(shapely.simplify(part, 1.0))).astype(np.int32)
            if len(co) < 2 or np.abs(co).max() > 32767: continue
            w.write(struct.pack('<BBH', keep[t], min(255, int(num(row[ix['Street Width']]))), len(co))); w.write(co.astype('<i2').tobytes())
