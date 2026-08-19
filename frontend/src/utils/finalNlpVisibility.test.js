import {
  applyCappedGlinerLabelsToTieredEntities,
  filterFinalNlpsForDisplay,
  filterHiddenGlinerLabelNlps,
  isExcludedWeakNlp,
  isHiddenGlinerLabelNlp,
  isHiddenFromFinalNlps,
  normalizeFinalNlpText,
  resolveVisibleTierEntities,
  stripLeadingWeakTokens,
} from './finalNlpVisibility';



describe('finalNlpVisibility', () => {

  test('isHiddenGlinerLabelNlp matches fully hidden labels only', () => {

    expect(isHiddenGlinerLabelNlp({ text: 'Poodle', label: 'Dog Breed' })).toBe(false);

    expect(isHiddenGlinerLabelNlp({ text: 'Purina', label: 'Brand' })).toBe(true);

    expect(isHiddenGlinerLabelNlp({ text: '123 Main St', label: 'Address' })).toBe(true);

    expect(isHiddenGlinerLabelNlp({ text: 'English', label: 'Language' })).toBe(true);

    expect(isHiddenGlinerLabelNlp({ text: 'Acme Corp', label: 'Company' })).toBe(true);

    expect(isHiddenGlinerLabelNlp({ text: 'Poodle', label: 'Pet' })).toBe(false);

    expect(isHiddenGlinerLabelNlp({ text: 'Poodle' })).toBe(false);

  });



  test('isExcludedWeakNlp matches single-token the, a, and pronouns only', () => {

    expect(isExcludedWeakNlp({ text: 'the' })).toBe(true);

    expect(isExcludedWeakNlp({ text: 'The' })).toBe(true);

    expect(isExcludedWeakNlp({ text: 'a' })).toBe(true);

    expect(isExcludedWeakNlp({ text: 'they' })).toBe(true);

    expect(isExcludedWeakNlp({ text: 'for' })).toBe(false);

    expect(isExcludedWeakNlp({ text: 'best for families' })).toBe(false);

    expect(isExcludedWeakNlp({ text: 'for the family' })).toBe(false);
    expect(isExcludedWeakNlp({ text: 'The Pug' })).toBe(false);
  });

  test('stripLeadingWeakTokens removes leading articles and pronouns from phrases', () => {
    expect(stripLeadingWeakTokens('The Pug')).toBe('Pug');
    expect(stripLeadingWeakTokens('The Shih Tzu')).toBe('Shih Tzu');
    expect(stripLeadingWeakTokens('Your Dog')).toBe('Dog');
    expect(stripLeadingWeakTokens('These Dogs')).toBe('Dogs');
    expect(stripLeadingWeakTokens('best for families')).toBe('best for families');
    expect(stripLeadingWeakTokens('for the family')).toBe('for the family');
    expect(normalizeFinalNlpText('the')).toBe('');
    expect(normalizeFinalNlpText('The')).toBe('');
  });

  test('filterHiddenGlinerLabelNlps removes fully hidden labels only', () => {

    const entities = [

      { text: 'Poodle', label: 'Dog Breed' },

      { text: 'Purina', label: 'Brand' },

      { text: '123 Main St', label: 'Address' },

      { text: 'English', label: 'Language' },

      { text: 'Acme Corp', label: 'Company' },

      { text: 'the', label: 'Other' },

      { text: 'family', label: 'Concept' },

    ];

    expect(filterHiddenGlinerLabelNlps(entities)).toEqual([

      { text: 'Poodle', label: 'Dog Breed' },

      { text: 'the', label: 'Other' },

      { text: 'family', label: 'Concept' },

    ]);

  });



  test('filterFinalNlpsForDisplay removes fully hidden labels and weak single tokens', () => {
    const entities = [
      { text: 'Poodle', label: 'Dog Breed' },
      { text: 'Acme Corp', label: 'Company' },
      { text: 'the', label: 'Other' },
      { text: 'they', label: 'Person' },
      { text: 'The Pug', label: 'Dog Breed', combined_count: 2 },
      { text: 'Pug', label: 'Dog Breed', combined_count: 1 },
      { text: 'for', label: 'Other' },
      { text: 'best for families', label: 'Phrase' },
      { text: 'family', label: 'Concept' },
    ];
    expect(filterFinalNlpsForDisplay(entities)).toEqual([
      { text: 'Poodle', label: 'Dog Breed' },
      { text: 'Pug', label: 'Dog Breed', combined_count: 2 },
      { text: 'for', label: 'Other' },
      { text: 'best for families', label: 'Phrase' },
      { text: 'family', label: 'Concept' },
    ]);
  });



  test('isHiddenFromFinalNlps combines label and weak-token rules', () => {

    expect(isHiddenFromFinalNlps({ text: 'Purina', label: 'Brand' })).toBe(true);

    expect(isHiddenFromFinalNlps({ text: 'the', label: 'Other' })).toBe(true);

    expect(isHiddenFromFinalNlps({ text: 'Poodle', label: 'Dog Breed' })).toBe(false);

    expect(isHiddenFromFinalNlps({ text: 'for', label: 'Other' })).toBe(false);

    expect(isHiddenFromFinalNlps({ text: 'best for families', label: 'Phrase' })).toBe(false);

  });



  test('applyCappedGlinerLabelsToTieredEntities keeps top 4 Dog Breed NLPs by tier order', () => {

    const result = applyCappedGlinerLabelsToTieredEntities({

      green: [

        { text: 'g1', label: 'Dog Breed' },

        { text: 'g2', label: 'Dog Breed' },

        { text: 'g3', label: 'Dog Breed' },

      ],

      orange: [

        { text: 'o1', label: 'Dog Breed' },

        { text: 'o2', label: 'Dog Breed' },

      ],

      white: [{ text: 'w1', label: 'Dog Breed' }],

    });



    const texts = [

      ...result.green,

      ...result.orange,

      ...result.white,

    ].map((e) => e.text);



    expect(texts).toEqual(['g1', 'g2', 'g3', 'o1']);

  });



  test('resolveVisibleTierEntities aligns tier counts with visible final nlps', () => {

    const mergeData = {

      green_nlps: [

        { text: 'Poodle', label: 'Dog Breed', combined_count: 3 },

        { text: 'the', label: 'Other', combined_count: 1 },

        { text: 'family', label: 'Concept', combined_count: 5 },

      ],

      orange_nlps: [{ text: 'kids', label: 'Person', combined_count: 2 }],

      white_nlps: [],

    };

    const sourceFiltered = [

      { text: 'Poodle', label: 'Dog Breed' },

      { text: 'the', label: 'Other' },

      { text: 'family', label: 'Concept' },

      { text: 'kids', label: 'Person' },

    ];



    const result = resolveVisibleTierEntities(mergeData, sourceFiltered, 'best family dog breeds');



    expect(result.highEntities).toHaveLength(2);

    expect(result.highEntities.map((e) => e.text)).toEqual(['Poodle', 'family']);

    expect(result.mediumEntities).toHaveLength(1);

    expect(result.visibleFinalNlps).toHaveLength(3);

    expect(result.visibleStats.uniqueCount).toBe(3);

    expect(result.visibleStats.occurrenceCount).toBe(10);

  });



  test('resolveVisibleTierEntities caps Dog Breed NLPs across tiers', () => {

    const mergeData = {

      green_nlps: [

        { text: 'breed-a', label: 'Dog Breed', combined_count: 5 },

        { text: 'breed-b', label: 'Dog Breed', combined_count: 4 },

        { text: 'family', label: 'Concept', combined_count: 3 },

      ],

      orange_nlps: [

        { text: 'breed-c', label: 'Dog Breed', combined_count: 2 },

        { text: 'breed-d', label: 'Dog Breed', combined_count: 1 },

        { text: 'breed-e', label: 'Dog Breed', combined_count: 1 },

      ],

      white_nlps: [{ text: 'breed-f', label: 'Dog Breed', combined_count: 1 }],

    };

    const sourceFiltered = [

      { text: 'breed-a', label: 'Dog Breed' },

      { text: 'breed-b', label: 'Dog Breed' },

      { text: 'family', label: 'Concept' },

      { text: 'breed-c', label: 'Dog Breed' },

      { text: 'breed-d', label: 'Dog Breed' },

      { text: 'breed-e', label: 'Dog Breed' },

      { text: 'breed-f', label: 'Dog Breed' },

    ];



    const result = resolveVisibleTierEntities(mergeData, sourceFiltered, 'best family dog breeds');

    const dogBreedTexts = result.visibleFinalNlps

      .filter((e) => e.label === 'Dog Breed')

      .map((e) => e.text);



    expect(dogBreedTexts).toEqual(['breed-a', 'breed-b', 'breed-c', 'breed-d']);

    expect(result.visibleFinalNlps).toHaveLength(5);

  });

});


