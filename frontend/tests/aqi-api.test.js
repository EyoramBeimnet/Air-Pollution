const test = require('node:test');
const assert = require('node:assert/strict');
const { normalizeAqiPayload, fetchAqiData } = require('../scripts/aqi-api.js');
const { FIPS, YEARS, validRows } = require('./fixtures.js');

const contract = { fips: FIPS, years: YEARS, lastObservedYear: 2025 };
const response = (body, options = {}) => ({ ok: options.ok ?? true, status: options.status ?? 200, json: async () => body });

test('accepts a valid array response and exposes its AQI values', () => {
  const data = normalizeAqiPayload(validRows(), contract);
  assert.equal(data['36005'][2020].aqi, 42);
  assert.equal(data['36085'][2030].forecast, true);
  assert.equal(data['36085'][2030].lo, 53);
});

test('accepts the supported { data: [...] } response format', () => {
  const data = normalizeAqiPayload({ data: validRows() }, contract);
  assert.equal(data['36061'][2027].aqi, 51);
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
  const rows = validRows().filter(row => !(row.fips === '36005' && row.year === 2027));
  assert.throws(() => normalizeAqiPayload(rows, contract), /36005\/2027/);
});

test('rejects invalid AQI values', () => {
  const rows = validRows();
  rows[0].aqi = -1;
  assert.throws(() => normalizeAqiPayload(rows, contract), /Invalid AQI value for 36005\/2020/);
});

test('rejects invalid forecast bounds', () => {
  const rows = validRows();
  const record = rows.find(row => row.forecast);
  record.lo = record.aqi + 1;
  assert.throws(() => normalizeAqiPayload(rows, contract), /Invalid forecast range/);
});
