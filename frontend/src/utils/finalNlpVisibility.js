import { resolveTierEntities } from './nlpTiers';



/** GLiNER entity-type labels fully hidden from final NLP display and counts. */

export const HIDDEN_GLINER_LABELS = new Set([

  'Brand',

  'Address',

  'Language',

  'Company',

]);



export const DOG_BREED_LABEL = 'Dog Breed';



/** GLiNER labels shown with a per-label NLP cap (global across tiers). */

export const CAPPED_GLINER_LABELS = {

  [DOG_BREED_LABEL]: 4,

};



/** Single-token NLPs excluded from final display (articles + pronouns). */

export const EXCLUDED_WEAK_NLP_TOKENS = new Set([

  'the',

  'a',

  'i',

  'me',

  'you',

  'he',

  'him',

  'she',

  'her',

  'it',

  'we',

  'us',

  'they',

  'them',

  'my',

  'your',

  'his',

  'hers',

  'our',

  'their',

  'mine',

  'yours',

  'ours',

  'theirs',

  'who',

  'whom',

  'whose',

  'that',

  'this',

  'these',

  'those',

]);



const TIER_RANK = { green: 0, orange: 1, white: 2 };



function entityKey(entity) {

  return (entity?.text || '').trim().toLowerCase();

}



function normalizeTokenWord(word) {
  return (word || '')
    .replace(/'/g, "'")
    .replace(/^["'()[]{}<>.,;:!?-]+|["'()[]{}<>.,;:!?-]+$/g, '')
    .split("'")[0]
    .toLowerCase();
}

function normalizeSingleToken(text) {
  const trimmed = (text || '').trim();
  if (!trimmed || trimmed.includes(' ')) {
    return null;
  }
  return normalizeTokenWord(trimmed);
}

/** Remove leading articles and pronouns from multi-word NLP text. */
export function stripLeadingWeakTokens(text) {
  const words = (text || '').trim().split(/\s+/).filter(Boolean);
  while (words.length > 0) {
    const token = normalizeTokenWord(words[0]);
    if (!EXCLUDED_WEAK_NLP_TOKENS.has(token)) {
      break;
    }
    words.shift();
  }
  return words.join(' ');
}

export function normalizeFinalNlpText(text) {
  const stripped = stripLeadingWeakTokens(text);
  if (!stripped) {
    return '';
  }
  const single = normalizeSingleToken(stripped);
  if (single && EXCLUDED_WEAK_NLP_TOKENS.has(single)) {
    return '';
  }
  return stripped;
}

export function normalizeEntityForFinalDisplay(entity) {
  if (!entity || isHiddenGlinerLabelNlp(entity)) {
    return null;
  }
  const original = (entity?.text || '').trim();
  if (!original) {
    return null;
  }
  const text = normalizeFinalNlpText(original);
  if (!text) {
    return null;
  }
  if (text === original) {
    return entity;
  }
  return { ...entity, text };
}



export function getEntityTierRank(entity, tierHint) {

  const fromEntity = (entity?.tier || '').toString().toLowerCase();

  const raw = fromEntity || (tierHint || '').toString().toLowerCase();

  if (raw in TIER_RANK) return TIER_RANK[raw];

  return TIER_RANK.orange;

}



export function capNlpsForLabel(entities, label, cap, tierLookup) {

  const matching = (entities || [])

    .map((entity, index) => ({ entity, index }))

    .filter(({ entity }) => (entity?.label || '').trim() === label);



  if (!matching.length || cap == null || matching.length <= cap) {

    return matching.map(({ entity }) => entity);

  }



  const sorted = [...matching].sort((a, b) => {

    const tierA = getEntityTierRank(

      a.entity,

      tierLookup?.get?.(entityKey(a.entity))

    );

    const tierB = getEntityTierRank(

      b.entity,

      tierLookup?.get?.(entityKey(b.entity))

    );

    if (tierA !== tierB) return tierA - tierB;

    return a.index - b.index;

  });



  return sorted.slice(0, cap).map(({ entity }) => entity);

}



export function applyCappedGlinerLabelsToTieredEntities({ green = [], orange = [], white = [] }) {

  const tagged = [

    ...green.map((entity, index) => ({ entity, tier: 'green', index })),

    ...orange.map((entity, index) => ({ entity, tier: 'orange', index })),

    ...white.map((entity, index) => ({ entity, tier: 'white', index })),

  ];



  const allowedByLabel = new Map();

  for (const [label, cap] of Object.entries(CAPPED_GLINER_LABELS)) {

    const labelEntities = tagged.filter(

      ({ entity }) => (entity?.label || '').trim() === label

    );

    const sorted = [...labelEntities].sort((a, b) => {

      const tierDiff =

        getEntityTierRank(a.entity, a.tier) - getEntityTierRank(b.entity, b.tier);

      if (tierDiff !== 0) return tierDiff;

      return a.index - b.index;

    });

    allowedByLabel.set(

      label,

      new Set(sorted.slice(0, cap).map(({ entity }) => entityKey(entity)))

    );

  }



  const keepEntity = (entity) => {

    const label = (entity?.label || '').trim();

    const allowed = allowedByLabel.get(label);

    if (!allowed) return true;

    return allowed.has(entityKey(entity));

  };



  return {

    green: green.filter(keepEntity),

    orange: orange.filter(keepEntity),

    white: white.filter(keepEntity),

  };

}



export function isHiddenGlinerLabelNlp(entity) {

  const label = (entity?.label || '').trim();

  return HIDDEN_GLINER_LABELS.has(label);

}



export function isExcludedWeakNlp(entity) {

  const token = normalizeSingleToken(entity?.text);

  if (!token) {

    return false;

  }

  return EXCLUDED_WEAK_NLP_TOKENS.has(token);

}



export function isHiddenFromFinalNlps(entity) {

  return isHiddenGlinerLabelNlp(entity) || isExcludedWeakNlp(entity);

}



export function filterHiddenGlinerLabelNlps(entities) {

  return (entities || []).filter((entity) => !isHiddenGlinerLabelNlp(entity));

}



export function filterFinalNlpsForDisplay(entities) {
  const byKey = new Map();
  for (const entity of entities || []) {
    const normalized = normalizeEntityForFinalDisplay(entity);
    if (!normalized) {
      continue;
    }
    const key = entityKey(normalized);
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, normalized);
      continue;
    }
    const existingCount = Number(existing.combined_count) || 0;
    const nextCount = Number(normalized.combined_count) || 0;
    if (nextCount > existingCount) {
      byKey.set(key, normalized);
    }
  }
  return Array.from(byKey.values());
}



function intersectWithSourceFilter(entities, visibleKeys) {

  if (!visibleKeys?.size) return entities || [];

  return (entities || []).filter((entity) => visibleKeys.has(entityKey(entity)));

}



export function countNlpsByTier(entities) {

  const counts = { green: 0, orange: 0, white: 0 };

  for (const entity of entities || []) {

    const tier = (entity.tier || '').toLowerCase();

    if (tier === 'green') counts.green += 1;

    else if (tier === 'white') counts.white += 1;

    else counts.orange += 1;

  }

  return counts;

}



export function resolveVisibleTierEntities(mergeData, sourceFilteredEntities, keyword) {

  const { highEntities, whiteEntities, mediumEntities } = resolveTierEntities(

    mergeData,

    sourceFilteredEntities,

    keyword

  );



  const visibleKeys = new Set((sourceFilteredEntities || []).map(entityKey));

  const usesPrecomputedTiers = Array.isArray(mergeData?.green_nlps);

  const shouldIntersect = usesPrecomputedTiers && visibleKeys.size > 0;



  const intersect = (entities) =>

    shouldIntersect ? intersectWithSourceFilter(entities, visibleKeys) : entities || [];



  const capped = applyCappedGlinerLabelsToTieredEntities({

    green: intersect(highEntities),

    orange: intersect(mediumEntities),

    white: intersect(whiteEntities),

  });



  const high = filterFinalNlpsForDisplay(capped.green);

  const medium = filterFinalNlpsForDisplay(capped.orange);

  const white = filterFinalNlpsForDisplay(capped.white);

  const visibleFinalNlps = [...high, ...medium, ...white];



  return {

    highEntities: high,

    mediumEntities: medium,

    whiteEntities: white,

    visibleFinalNlps,

    visibleStats: {

      uniqueCount: visibleFinalNlps.length,

      occurrenceCount: visibleFinalNlps.reduce(

        (sum, entity) => sum + (entity.combined_count || 0),

        0

      ),

    },

  };

}


