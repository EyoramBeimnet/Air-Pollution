const test = require('node:test');
const assert = require('node:assert/strict');
const { normalizeAqiPayload, fetchAqiData, normalizeMonthlyAqiPayload } = require('../scripts/aqi-api.js');
const { FIPS, YEARS, validRows } = require('./fixtures.js');

const contract = { fips: FIPS, years: YEARS, lastObservedYear: 2025 };
const response = (body, options = {}) => ({ ok: options.ok ?? true, status: options.status ?? 200, json: async () => body });

test('accepts a valid array response and exposes its AQI values', () => {
  const data = normalizeAqiPayload(validRows(), contract);
  assert.equal(data['36005'][2020].aqi, 42);
  assert.equal(data['36085'][2025].forecast, false);
});

test('accepts the supported { data: [...] } response format', () => {
  const data = normalizeAqiPayload({ data: validRows() }, contract);
  assert.equal(data['36061'][2025].aqi, 49);
});

test('keeps a complete 2026 model forecast distinct from observations', () => {
  const data = normalizeAqiPayload(validRows(), contract);
  assert.equal(data['36005'][2025].forecast, false);
  assert.equal(data['36005'][2026].forecast, true);
});

test('rejects an unavailable API request', async () => {
  await assert.rejects(() => fetchAqiData('/api/aqi', contract, async () => { throw new TypeError('network down'); }), /network down/);
});

test('rejects an HTTP error response', async () => {
  await assert.rejects(() => fetchAqiData('/api/aqi', contract, async () => response({}, { ok: false, status: 503 })), /HTTP 503/);
});

test('rejects malformed JSON', async () => {
  const malformed = { ok: true, status: 200, json: async () => { throw new SyntaxError('bad JSON'); } };
  await assert.rejects(() => fetchAqiData('/api/aqi', contract, async () => malformed), /malformed JSON/);
});

test('rejects a missing borough', () => {
  const rows = validRows().filter(row => row.fips !== '36085');
  assert.throws(() => normalizeAqiPayload(rows, contract), /36085\/2020/);
});

test('rejects a missing year', () => {
  const rows = validRows().filter(row => !(row.fips === '36005' && row.year === 2025));
  assert.throws(() => normalizeAqiPayload(rows, contract), /36005\/2025/);
});

test('rejects invalid AQI values', () => {
  const rows = validRows();
  rows[0].aqi = -1;
  assert.throws(() => normalizeAqiPayload(rows, contract), /Invalid AQI value for 36005\/2020/);
});

test('rejects invalid forecast bounds', () => {
  const rows = validRows();
  const record = rows.find(row => row.year === 2025);
  record.forecast = true;
  record.lo = record.aqi + 1;
  assert.throws(() => normalizeAqiPayload(rows, contract), /Invalid forecast range/);
});

test('validates monthly observations and recursive forecasts separately', () => {
  const months = ['2025-12', '2026-01'];
  const rows = FIPS.flatMap((fips, index) => months.map(month => ({ fips, month, aqi: 40 + index, forecast: month > '2025-12' })));
  const data = normalizeMonthlyAqiPayload({ data: rows }, { fips: FIPS, months, lastObservedMonth: '2025-12' });
  assert.equal(data['36005']['2025-12'].forecast, false);
  assert.equal(data['36005']['2026-01'].forecast, true);
});
