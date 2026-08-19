import {
  countLabelsWithNlps,
  groupFinalNlpsByGlinerLabel,
  resolveGlinerLabel,
} from './glinerLabels';

describe('glinerLabels utils', () => {
  const catalog = ['Dog Breed', 'Brand', 'Address', 'Language', 'Company', 'City', 'Other'];

  test('resolveGlinerLabel maps unknown labels to Other', () => {
    expect(resolveGlinerLabel('Dog Breed', catalog)).toBe('Dog Breed');
    expect(resolveGlinerLabel('Brand', catalog)).toBe('Brand');
    expect(resolveGlinerLabel('Company', catalog)).toBe('Company');
    expect(resolveGlinerLabel('NLP', catalog)).toBe('Other');
    expect(resolveGlinerLabel('', catalog)).toBe('Other');
  });

  test('groupFinalNlpsByGlinerLabel returns visible labels except fully hidden ones', () => {
    const entities = [
      { text: 'Poodle', label: 'Dog Breed' },
      { text: 'Purina', label: 'Brand' },
      { text: '123 Main St', label: 'Address' },
      { text: 'English', label: 'Language' },
      { text: 'Acme Corp', label: 'Company' },
      { text: 'the', label: 'Other' },
      { text: 'Austin', label: 'City' },
      { text: 'legacy', label: 'NLP' },
    ];
    const grouped = groupFinalNlpsByGlinerLabel({ glinerLabels: catalog, entities });
    expect(grouped).toHaveLength(3);
    expect(grouped.find((row) => row.label === 'Dog Breed').nlps).toHaveLength(1);
    expect(grouped.find((row) => row.label === 'Brand')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'Address')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'Language')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'Company')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'City').nlps).toHaveLength(1);
    expect(grouped.find((row) => row.label === 'Other').nlps).toHaveLength(1);
  });

  test('groupFinalNlpsByGlinerLabel uses preGrouped when provided', () => {
    const grouped = groupFinalNlpsByGlinerLabel({
      glinerLabels: catalog,
      entities: [],
      preGrouped: {
        'Dog Breed': [{ text: 'Beagle' }],
        Brand: [{ text: 'Purina' }],
        Company: [{ text: 'Acme Corp' }],
        City: [{ text: 'Austin' }],
        Other: [{ text: 'the' }],
      },
    });
    expect(grouped.find((row) => row.label === 'Dog Breed').nlps[0].text).toBe('Beagle');
    expect(grouped.find((row) => row.label === 'Brand')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'Company')).toBeUndefined();
    expect(grouped.find((row) => row.label === 'City').nlps[0].text).toBe('Austin');
    expect(grouped.find((row) => row.label === 'City').nlps).toHaveLength(1);
    expect(grouped.find((row) => row.label === 'Other').nlps).toHaveLength(0);
  });

  test('groupFinalNlpsByGlinerLabel caps Dog Breed NLPs to top 4 by tier priority', () => {
    const mergeData = {
      green_nlps: [
        { text: 'breed-a', label: 'Dog Breed' },
        { text: 'breed-b', label: 'Dog Breed' },
        { text: 'breed-c', label: 'Dog Breed' },
      ],
      orange_nlps: [
        { text: 'breed-d', label: 'Dog Breed' },
        { text: 'breed-e', label: 'Dog Breed' },
      ],
      white_nlps: [{ text: 'breed-f', label: 'Dog Breed' }],
    };
    const grouped = groupFinalNlpsByGlinerLabel({
      glinerLabels: catalog,
      entities: [],
      preGrouped: {
        'Dog Breed': [
          { text: 'breed-a', label: 'Dog Breed' },
          { text: 'breed-b', label: 'Dog Breed' },
          { text: 'breed-c', label: 'Dog Breed' },
          { text: 'breed-d', label: 'Dog Breed' },
          { text: 'breed-e', label: 'Dog Breed' },
          { text: 'breed-f', label: 'Dog Breed' },
        ],
      },
      mergeData,
    });

    const dogBreedRow = grouped.find((row) => row.label === 'Dog Breed');
    expect(dogBreedRow.nlps.map((e) => e.text)).toEqual([
      'breed-a',
      'breed-b',
      'breed-c',
      'breed-d',
    ]);
  });

  test('countLabelsWithNlps counts populated labels', () => {
    const grouped = [
      { label: 'Dog Breed', nlps: [{ text: 'a' }] },
      { label: 'City', nlps: [] },
    ];
    expect(countLabelsWithNlps(grouped)).toEqual({ populated: 1, total: 2 });
  });
});
