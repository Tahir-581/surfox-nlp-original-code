const TOKEN_PATTERN = /[a-zA-Z]+/g;

const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'have',
  'he', 'her', 'his', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
  'their', 'them', 'they', 'this', 'to', 'was', 'were', 'with', 'you', 'your',
]);

const MIN_SUBSTRING_WORD_LEN = 3;
const MIN_NLPS_PER_BUCKET = 2;

const IRREGULAR_PLURALS = {
  breeds: 'breed',
  dogs: 'dog',
  cats: 'cat',
  people: 'person',
  children: 'child',
  geese: 'goose',
  teeth: 'tooth',
  feet: 'foot',
  mice: 'mouse',
  shelters: 'shelter',
  families: 'family',
  puppies: 'puppy',
  kittens: 'kitten',
  boxes: 'box',
  foxes: 'fox',
  bushes: 'bush',
  glasses: 'glass',
  wolves: 'wolf',
  lives: 'life',
  wives: 'wife',
  knives: 'knife',
  tzus: 'tzu',
  frises: 'frise',
  collies: 'collie',
  poodles: 'poodle',
  beagles: 'beagle',
  poodle: 'poodle',
  homes: 'home',
  times: 'time',
};

function singularizeWord(word) {
  const w = (word || '').toLowerCase().trim();
  if (!w) return w;
  if (IRREGULAR_PLURALS[w]) return IRREGULAR_PLURALS[w];
  if (w.endsWith('ies') && w.length > 3) return `${w.slice(0, -3)}y`;
  if (w.endsWith('zes') && w.length > 3) return w.slice(0, -2);
  if (w.endsWith('ses') && w.length > 3) return w.slice(0, -2);
  if ((w.endsWith('xes') || w.endsWith('ches') || w.endsWith('shes')) && w.length > 3) {
    return w.slice(0, -2);
  }
  if (w.endsWith('ves') && w.length > 3) return `${w.slice(0, -3)}f`;
  if (w.endsWith('oes') && w.length > 3) return w.slice(0, -2);
  if (w.endsWith('s') && w.length > 2) {
    if (!'aeiou'.includes(w[w.length - 2]) && !['ss', 'us', 'is'].includes(w.slice(-2))) {
      return w.slice(0, -1);
    }
  }
  return w;
}

function entityText(entity) {
  if (entity && typeof entity === 'object') {
    return (entity.text || '').trim();
  }
  return String(entity || '').trim();
}

function tokenizeNlp(text) {
  const matches = (text || '').toLowerCase().match(TOKEN_PATTERN);
  return matches || [];
}

function normalizeToken(token) {
  return singularizeWord((token || '').toLowerCase().trim());
}

function collectBucketWords(entities) {
  const words = new Set();
  for (const entity of entities) {
    const text = entityText(entity);
    if (!text) continue;
    const tokens = tokenizeNlp(text);
    if (!tokens.length) continue;
    if (tokens.length === 1) {
      const normalized = normalizeToken(tokens[0]);
      if (normalized && !STOP_WORDS.has(normalized)) words.add(normalized);
    }
    for (const token of tokens) {
      const normalized = normalizeToken(token);
      if (normalized && !STOP_WORDS.has(normalized)) words.add(normalized);
    }
  }
  return words;
}

function nlpMatchesWord(nlpText, word) {
  if (!nlpText || !word) return false;
  const tokens = tokenizeNlp(nlpText);
  if (!tokens.length) return false;

  if (tokens.length === 1) {
    const single = normalizeToken(tokens[0]);
    if (single === word) return true;
  }

  for (const token of tokens) {
    const normalized = normalizeToken(token);
    if (normalized === word) return true;
    if (word.length >= MIN_SUBSTRING_WORD_LEN && normalized.includes(word)) return true;
  }

  return false;
}

export function buildWordBuckets(entities = []) {
  const entityList = (entities || []).filter((e) => entityText(e));
  if (!entityList.length) return [];

  const bucketWords = collectBucketWords(entityList);
  const buckets = [];

  for (const word of bucketWords) {
    const matching = [];
    const seenTexts = new Set();

    for (const entity of entityList) {
      const text = entityText(entity);
      const key = text.toLowerCase();
      if (seenTexts.has(key)) continue;
      if (!nlpMatchesWord(text, word)) continue;
      seenTexts.add(key);
      const combined = Number(entity?.combined_count) || 0;
      matching.push({ text, combined_count: combined });
    }

    if (matching.length < MIN_NLPS_PER_BUCKET) continue;
    buckets.push({ word, nlp_count: matching.length, nlps: matching });
  }

  buckets.sort((a, b) => {
    if (b.nlp_count !== a.nlp_count) return b.nlp_count - a.nlp_count;
    return a.word.localeCompare(b.word);
  });

  return buckets;
}

export function buildTierWordBuckets(green = [], white = [], orange = []) {
  return {
    green: buildWordBuckets(green),
    white: buildWordBuckets(white),
    orange: buildWordBuckets(orange),
  };
}

export function resolveWordBuckets(mergeData) {
  if (mergeData?.word_buckets) return mergeData.word_buckets;
  return buildTierWordBuckets(
    mergeData?.green_nlps,
    mergeData?.white_nlps,
    mergeData?.orange_nlps
  );
}
