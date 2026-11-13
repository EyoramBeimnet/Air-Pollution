/*
 * Browser-side AQI API contract adapter.
 * This file intentionally contains no application-specific rendering code so
 * the same validation rules can be tested with Node's built-in test runner.
 */
(function exposeAqiApi(root) {
  function normalizeAqiPayload(payload, { fips, years, lastObservedYear }) {
    const rows = Array.isArray(payload) ? payload : payload?.data;
    if (!Array.isArray(rows)) throw new Error('Expected an array or { data: [...] }.');

    const out = Object.fromEntries(fips.map(code => [code, {}]));
    for (const row of rows) {
      const code = String(row.fips ?? '').padStart(5, '0');
      const year = Number(row.year);
      const aqi = Number(row.aqi);
      if (!out[code] || !years.includes(year)) continue;
      if (!Number.isFinite(aqi) || aqi < 0) throw new Error(`Invalid AQI value for ${code}/${year}.`);

      const forecast = row.forecast ?? year > lastObservedYear;
      const lo = row.lo == null ? undefined : Number(row.lo);
      const hi = row.hi == null ? undefined : Number(row.hi);
      if (forecast && (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > aqi || aqi > hi)) {
        throw new Error(`Invalid forecast range for ${code}/${year}.`);
      }
      out[code][year] = { aqi, forecast: Boolean(forecast), ...(forecast ? { lo, hi } : {}) };
    }

    const missing = fips.flatMap(code => years.filter(year => !out[code][year]).map(year => `${code}/${year}`));
    if (missing.length) throw new Error(`Missing required AQI records: ${missing.join(', ')}.`);
    return out;
  }

  async function fetchAqiData(url, contract, fetchImpl = fetch) {
    const response = await fetchImpl(url, { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error('API returned malformed JSON.');
    }
    return normalizeAqiPayload(payload, contract);
  }

  const api = { normalizeAqiPayload, fetchAqiData };
  root.AirshedApi = api;
  if (typeof module !== 'undefined') module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
