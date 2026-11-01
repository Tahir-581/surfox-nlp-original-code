export function filterSurferTerms(entities, { includedOnly = false, isNlpOnly = false } = {}) {
  const list = Array.isArray(entities) ? entities : [];
  return list.filter((entity) => {
    if (includedOnly && !entity.included) return false;
    if (isNlpOnly && !entity.is_nlp) return false;
    return true;
  });
}

export function resolveSurferTierEntities(mergeView, filteredEntities) {
  const filtered = Array.isArray(filteredEntities) ? filteredEntities : [];
  if (filtered.length === 0) {
    return { highEntities: [], whiteEntities: [], mediumEntities: [] };
  }

  const visible = new Set(filtered.map((e) => e.text));
  const green = (mergeView?.green_nlps || []).filter((e) => visible.has(e.text));
  const white = (mergeView?.white_nlps || []).filter((e) => visible.has(e.text));

  if (green.length > 0 || white.length > 0) {
    return {
      highEntities: green,
      whiteEntities: white,
      mediumEntities: [],
    };
  }

  return {
    highEntities: filtered.filter((e) => e.included),
    whiteEntities: filtered.filter((e) => !e.included),
    mediumEntities: [],
  };
}
