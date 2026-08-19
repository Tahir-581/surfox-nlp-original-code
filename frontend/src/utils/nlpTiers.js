const NUMERICAL_NLP_PATTERN = /\d/;
const GREEN_TIER_RATIO = 0.35;
const WHITE_TIER_RATIO_OF_REMAINDER = 0.1;

export function isNumericalNlp(text) {
  return NUMERICAL_NLP_PATTERN.test(text || '');
}

export function isExemptNumericalNlp(text, exemptTexts = []) {
  const normalized = (text || '').trim().toLowerCase();
  if (!normalized) return false;
  const exempt = new Set(
    exemptTexts.filter(Boolean).map((t) => t.trim().toLowerCase())
  );
  return exempt.has(normalized);
}

function itemText(item) {
  if (item && typeof item === 'object') {
    return (item.text || '').trim();
  }
  return String(item || '').trim();
}

export function capNumericalNlps(items, { maxNumerical = 2, exemptTexts = [] } = {}) {
  const kept = [];
  let numericalCount = 0;

  for (const item of items) {
    const text = itemText(item);
    if (!text) continue;

    if (isNumericalNlp(text) && !isExemptNumericalNlp(text, exemptTexts)) {
      if (numericalCount >= maxNumerical) continue;
      numericalCount += 1;
    }
    kept.push(item);
  }

  return kept;
}

export function splitEntitiesIntoTiers(
  entities,
  { maxNumericalPerTier = 2, exemptTexts = [] } = {}
) {
  const sorted = [...entities].sort(
    (a, b) => (Number(b.average_weightage) || 0) - (Number(a.average_weightage) || 0)
  );
  const total = sorted.length;
  if (total === 0) {
    return { green: [], white: [], orange: [] };
  }

  const greenCount = Math.max(1, Math.floor(total * GREEN_TIER_RATIO));
  const greenRaw = sorted.slice(0, greenCount);
  const remaining = sorted.slice(greenCount);
  const whiteCount =
    remaining.length > 0
      ? Math.max(1, Math.floor(remaining.length * WHITE_TIER_RATIO_OF_REMAINDER))
      : 0;
  const whiteRaw = remaining.slice(0, whiteCount);
  const orangeRaw = remaining.slice(whiteCount);

  return {
    green: capNumericalNlps(greenRaw, { maxNumerical: maxNumericalPerTier, exemptTexts }),
    white: capNumericalNlps(whiteRaw, { maxNumerical: maxNumericalPerTier, exemptTexts }),
    orange: capNumericalNlps(orangeRaw, { maxNumerical: maxNumericalPerTier, exemptTexts }),
  };
}

export function resolveTierEntities(
  mergeData,
  entities,
  keyword,
  { maxNumericalPerTier = 2 } = {}
) {
  if (Array.isArray(mergeData?.green_nlps)) {
    return {
      highEntities: mergeData.green_nlps || [],
      whiteEntities: mergeData.white_nlps || [],
      mediumEntities: mergeData.orange_nlps || [],
    };
  }

  const exemptTexts = keyword ? [keyword] : [];
  const tiers = splitEntitiesIntoTiers(entities, { maxNumericalPerTier, exemptTexts });
  return {
    highEntities: tiers.green,
    whiteEntities: tiers.white,
    mediumEntities: tiers.orange,
  };
}
