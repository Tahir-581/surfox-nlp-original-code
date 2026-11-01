import {
  CAPPED_GLINER_LABELS,
  capNlpsForLabel,
  filterFinalNlpsForDisplay,
  HIDDEN_GLINER_LABELS,
} from './finalNlpVisibility';

const TIER_CLASS = {
  green: 'tier-high',
  orange: 'tier-medium',
  white: 'tier-low',
};

export function resolveGlinerLabel(entityLabel, catalog) {
  const label = (entityLabel || '').trim();
  if (catalog.includes(label)) return label;
  return 'Other';
}

export function buildTierLookup(mergeData) {
  const map = new Map();
  const tiers = [
    ['green', 'green_nlps'],
    ['orange', 'orange_nlps'],
    ['white', 'white_nlps'],
  ];
  for (const [tier, key] of tiers) {
    for (const entity of mergeData?.[key] || []) {
      const text = (entity.text || '').trim().toLowerCase();
      if (text) map.set(text, tier);
    }
  }
  return map;
}

export function tierClassForEntity(entity, tierLookup) {
  const text = (entity.text || '').trim().toLowerCase();
  const tier = tierLookup.get(text) || 'orange';
  return TIER_CLASS[tier] || TIER_CLASS.orange;
}

function processLabelNlps(label, nlps, tierLookup) {
  let list = (Array.isArray(nlps) ? nlps : []).map((entity) => ({
    ...entity,
    label: (entity?.label || '').trim() || label,
  }));
  const cap = CAPPED_GLINER_LABELS[label];
  if (cap != null) {
    list = capNlpsForLabel(list, label, cap, tierLookup);
  }
  return filterFinalNlpsForDisplay(list);
}

export function groupFinalNlpsByGlinerLabel({
  glinerLabels,
  entities,
  preGrouped,
  mergeData,
}) {
  const catalog = Array.isArray(glinerLabels) && glinerLabels.length
    ? [...glinerLabels]
    : [];
  const tierLookup = buildTierLookup(mergeData || {});

  if (preGrouped && typeof preGrouped === 'object' && catalog.length) {
    return catalog
      .filter((label) => !HIDDEN_GLINER_LABELS.has(label))
      .map((label) => ({
        label,
        nlps: processLabelNlps(
          label,
          Array.isArray(preGrouped[label]) ? preGrouped[label] : [],
          tierLookup
        ),
      }));
  }

  const buckets = Object.fromEntries(catalog.map((label) => [label, []]));
  const seen = Object.fromEntries(catalog.map((label) => [label, new Set()]));

  for (const entity of entities || []) {
    const text = (entity.text || '').trim();
    if (!text) continue;
    const bucketLabel = resolveGlinerLabel(entity.label, catalog);
    const key = text.toLowerCase();
    if (seen[bucketLabel]?.has(key)) continue;
    seen[bucketLabel]?.add(key);
    buckets[bucketLabel].push(entity);
  }

  return catalog
    .filter((label) => !HIDDEN_GLINER_LABELS.has(label))
    .map((label) => ({
      label,
      nlps: processLabelNlps(label, buckets[label] || [], tierLookup),
    }));
}

export function countLabelsWithNlps(grouped) {
  const populated = (grouped || []).filter((row) => row.nlps.length > 0).length;
  return { populated, total: (grouped || []).length };
}
