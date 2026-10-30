"""Pack NYC building footprints (NYC Open Data 5zhs-2jue, CSV export) into compact binaries.

Usage: python process_buildings.py export.csv OUT_DIR
Writes hero.bin (full outlines: all Manhattan, anything >=15 m, or >=6000 m2) and
lo_{2..5}.bin (oriented boxes for the rest, per borough). Coordinates are metres from
(-73.975, 40.703). The site loads them base64-encoded: `base64 -w0 hero.bin > hero.txt`.
"""
import csv, sys, struct, time, json, os
import numpy as np, shapely
csv.field_size_limit(10**9)
LON0, LAT0 = -73.975, 40.703
COS = np.cos(np.radians(LAT0)); MPD = 111320.0
src, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
def proj(c):
    c = c.copy(); c[:,0] = (c[:,0]-LON0)*COS*MPD; c[:,1] = (c[:,1]-LAT0)*MPD; return c
def num(s):
    try: return float(s.replace(',',''))
    except: return 0.0
hero = open(os.path.join(OUT,'hero.bin'),'wb')
lo = {b: open(os.path.join(OUT,f'lo_{b}.bin'),'wb') for b in '12345'}
stats = {'hero':0,'lo':{b:0 for b in '12345'},'skipped':0}
r = csv.reader(open(src, newline='', encoding='utf-8')); h = next(r); ix = {k:i for i,k in enumerate(h)}
def flush(rows):
    g = shapely.from_wkt(np.array([x[0] for x in rows], dtype=object), on_invalid='ignore')
    parts, idx = shapely.get_parts(g, return_index=True)
    pa = shapely.area(parts)
    order = np.lexsort((-pa, idx)); first = np.ones(len(order), bool); first[1:] = idx[order][1:] != idx[order][:-1]
    sel = order[first]; big = parts[sel]; bidx = idx[sel]
    ring = shapely.transform(shapely.polygons(shapely.get_exterior_ring(big)), proj)
    area = shapely.area(ring); cen = shapely.get_coordinates(shapely.centroid(ring))
    for k, gi in enumerate(bidx):
        _, hft, bbl, fc = rows[gi]; b = bbl[:1]
        if b not in '12345' or fc in ('5,110','2,110','1,003') or area[k] < 12: stats['skipped'] += 1; continue
        hm = hft*0.3048 if hft > 0 else 8.0; cx, cy = cen[k]
        if b == '1' or hm >= 15 or area[k] >= 6000:
            co = shapely.get_coordinates(shapely.simplify(ring[k], 0.4))[:-1]
            if len(co) < 3: co = shapely.get_coordinates(ring[k])[:-1]
            if len(co) > 400: co = co[::int(np.ceil(len(co)/400))]
            icx, icy = int(round(cx)), int(round(cy))
            d = np.round((co - [icx, icy]) * 10).astype(np.int32)
            if np.abs(d).max() > 32767: stats['skipped'] += 1; continue
            hero.write(struct.pack('<hhHH', icx, icy, len(co), min(65535, int(round(hm*10))))); hero.write(d.astype('<i2').tobytes())
            stats['hero'] += 1
        else:
            c = shapely.get_coordinates(shapely.minimum_rotated_rectangle(ring[k]))
            if len(c) < 4: continue
            e1 = c[1]-c[0]; e2 = c[2]-c[1]; L1 = np.hypot(*e1); L2 = np.hypot(*e2)
            if L1 < L2: e1, L1, L2 = e2, L2, L1
            ang = np.degrees(np.arctan2(e1[1], e1[0])) % 180
            lo[b].write(struct.pack('<hhBBBB', int(round(cx)), int(round(cy)), min(255,int(round(L1*2))), min(255,int(round(L2*2))), int(round(ang/180*255))%256, min(255,int(round(hm*4)))))
            stats['lo'][b] += 1
rows = []
for row in r:
    rows.append((row[ix['the_geom']], num(row[ix['Height Roof']]), row[ix['BASE_BBL']], row[ix['Feature Code']]))
    if len(rows) == 40000: flush(rows); rows = []
if rows: flush(rows)
hero.close(); [x.close() for x in lo.values()]
print(json.dumps(stats))
