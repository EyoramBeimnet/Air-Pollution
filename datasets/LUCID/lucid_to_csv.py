#!/usr/bin/env python3
# LUCID Dataset Converter

"""
lucid_to_csv.py

Converts a LUCID "Total Surface Area by broad cover/use and by year" export (fixed-width txt file) into a CSV
where each land-use category is expressed as a percentage of that row's total area, instead of raw acreage.

The Total column is dropped from the output, since it isn't useful in modeling.

The parser does not hardcode the list of land-use categories. 
It reads the category names directly from the "Broad Cover/Use =" filter line in the
file's header, so it adapts automatically if a different set/order of
categories was selected in LUCID. 

It also supports files that contain more than one State/County/Year block 
(e.g. an export covering several counties).



Usage:
    python lucid_to_csv.py input output
    python lucid_to_csv.py input
    (no need to type .txt / .csv — they're added automatically; if you omit
    the output name, the CSV is written using the same base name as the
    input file)

"""


import argparse
import csv
import re
import sys
 
 
# Matches a metadata block: state name, county name, and year range.
# LUCID prints these as e.g.:
#   State Name: New York (36)   | County Name: Bronx (36005)   | Years: 2019 - 2022 |
META_RE = re.compile(
    r"State Name:\s*([^|(]+?)\s*\(\d+\)\s*\|\s*"
    r"County Name:\s*([^|(]+?)\s*\(\d+\)\s*\|\s*"
    r"Years:\s*(\d{4})\s*-\s*(\d{4})"
)
 
# Matches the category list inside the FILTERS line, e.g.:
#   Broad Cover/Use =Cultivated Cropland-1  Noncultivated Cropland-2 ...
CATEGORY_BLOCK_RE = re.compile(r"Broad Cover/Use\s*=(.+?)\s*\|\s*Geo Level")
 
# A data row starts with a 4-digit year, then whitespace-separated numbers.
# The margin-of-error row underneath (in parentheses) is intentionally NOT
# matched by this pattern, since it doesn't start with a bare year.
DATA_ROW_RE = re.compile(r"^\s*(\d{4})\s+([\d,.\s]+?)\s*$")
 
NUMBER_RE = re.compile(r"[\d,]+\.\d+")
 
 
def parse_categories(text):
    """Extract the ordered list of land-use category names from the
    'Broad Cover/Use =' filter line. Returns [] if not found."""
    match = CATEGORY_BLOCK_RE.search(text)
    if not match:
        return []
    raw = match.group(1).strip()
    # Categories are separated by 2+ spaces, each token ending in "-<digit>".
    tokens = re.split(r"\s{2,}", raw)
    categories = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        name = re.sub(r"-\d+$", "", tok).strip()
        if name:
            categories.append(name)
    return categories
 
 
def parse_data_rows(text, num_categories):
    # Find every row that starts with a 4-digit year followed by exactly
    # num_categories + 1 (categories + total) numeric values. Returns a list
    # of (year, [values...]) tuples, in the order they appear.
    rows = []
    for line in text.splitlines():
        match = DATA_ROW_RE.match(line)
        if not match:
            continue
        year = match.group(1)
        numbers = NUMBER_RE.findall(match.group(2))
        if len(numbers) != num_categories + 1:
            # Not a data row with the expected column count (could be a
            # stray line, a totals footer, etc.) — skip it.
            continue
        values = [float(n.replace(",", "")) for n in numbers]
        rows.append((year, values))
    return rows
 
 
def parse_lucid_export(text):
    # Parse a full LUCID export (possibly containing several State/County
    # blocks) into a list of dicts, one per County-Year row, with each
    # category expressed as a percentage of that row's total area.
    categories = parse_categories(text)
    if not categories:
        raise ValueError(
            "Could not find the 'Broad Cover/Use =' category list in the "
            "input file. Make sure this is an unmodified LUCID export."
        )
 
    meta_matches = list(META_RE.finditer(text))
    if not meta_matches:
        raise ValueError(
            "Could not find any 'State Name / County Name / Years' header "
            "in the input file."
        )
 
    results = []
    for i, meta in enumerate(meta_matches):
        state, county, year_start, year_end = meta.groups()
        block_start = meta.end()
        block_end = meta_matches[i + 1].start() if i + 1 < len(meta_matches) else len(text)
        block_text = text[block_start:block_end]
 
        data_rows = parse_data_rows(block_text, len(categories))
        for year, values in data_rows:
            *category_values, total = values
            row = {
                "State": state.strip(),
                "County": county.strip(),
                "Year": year,
            }
            for cat_name, cat_value in zip(categories, category_values):
                if total > 0:
                    pct = round(cat_value / total * 100, 4)
                else:
                    pct = 0.0
                row[cat_name] = pct
            results.append(row)
 
    return categories, results
 
 
def write_csv(categories, rows, output_path):
    fieldnames = ["State", "County", "Year"] + categories
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
 
 
def ensure_extension(path, extension):
    # Append extension (e.g. '.txt') if path doesn't already end with it,
    # so the user can pass a bare filename without typing the extension.
    if path.lower().endswith(extension):
        return path
    return path + extension
 
 
def strip_extension(path, extension):
    # Remove a known extension from path, if present, so it can be reused
    # as the base name for a different extension.
    if path.lower().endswith(extension):
        return path[: -len(extension)]
    return path
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Convert a LUCID land-use export (.txt) into a percentage-of-total CSV."
    )
    parser.add_argument("input", help="Path to the LUCID exported table (.txt assumed)")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to write the output (.csv assumed). If omitted, uses the input's base name.",
    )
    args = parser.parse_args()
 
    input_path = ensure_extension(args.input, ".txt")
 
    if args.output is None:
        base_name = strip_extension(args.input, ".txt")
        output_path = ensure_extension(base_name, ".csv")
    else:
        output_path = ensure_extension(args.output, ".csv")
 
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
 
    categories, rows = parse_lucid_export(text)
    if not rows:
        print("No data rows were found — double-check the input file.", file=sys.stderr)
        sys.exit(1)
 
    write_csv(categories, rows, output_path)
    print(f"Wrote {len(rows)} row(s) covering {len(categories)} land-use categories to {output_path}")
 
 
if __name__ == "__main__":
    main()
 