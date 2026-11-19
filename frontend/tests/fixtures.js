const FIPS = ['36005', '36047', '36061', '36081', '36085'];
const YEARS = Array.from({ length: 11 }, (_, index) => 2020 + index);

function validRows() {
  return FIPS.flatMap((fips, boroughIndex) => YEARS.map(year => {
    const aqi = 42 + boroughIndex + (year - 2020);
    const forecast = year > 2025;
    return { fips, year, aqi, forecast, ...(forecast ? { lo: aqi - 3, hi: aqi + 3 } : {}) };
  }));
}

module.exports = { FIPS, YEARS, validRows };
