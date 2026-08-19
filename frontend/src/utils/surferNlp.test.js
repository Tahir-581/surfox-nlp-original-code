import {
  filterSurferTerms,
  resolveSurferTierEntities,
} from './surferNlp';

describe('surferNlp utils', () => {
  const entities = [
    { text: 'dog breeds', included: true, is_nlp: true },
    { text: 'apartments', included: true, is_nlp: false },
    { text: 'high energy', included: false, is_nlp: true },
  ];

  const mergeView = {
    green_nlps: [entities[0], entities[1]],
    white_nlps: [entities[2]],
    orange_nlps: [],
  };

  test('filterSurferTerms respects included and is_nlp toggles', () => {
    expect(filterSurferTerms(entities, {})).toHaveLength(3);
    expect(filterSurferTerms(entities, { includedOnly: true })).toHaveLength(2);
    expect(filterSurferTerms(entities, { isNlpOnly: true })).toHaveLength(2);
    expect(
      filterSurferTerms(entities, { includedOnly: true, isNlpOnly: true })
    ).toHaveLength(1);
  });

  test('resolveSurferTierEntities splits green and white from merge view', () => {
    const filtered = filterSurferTerms(entities, { isNlpOnly: true });
    const tiers = resolveSurferTierEntities(mergeView, filtered);
    expect(tiers.highEntities.map((e) => e.text)).toEqual(['dog breeds']);
    expect(tiers.whiteEntities.map((e) => e.text)).toEqual(['high energy']);
    expect(tiers.mediumEntities).toEqual([]);
  });
});
