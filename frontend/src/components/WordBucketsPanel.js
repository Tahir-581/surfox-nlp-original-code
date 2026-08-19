import React, { useMemo, useState } from 'react';
import { resolveWordBuckets } from '../utils/nlpWordBuckets';

const TIER_CONFIG = [
  { key: 'green', label: 'Green', className: 'tier-high' },
  { key: 'orange', label: 'Orange', className: 'tier-medium' },
  { key: 'white', label: 'White', className: 'tier-low' },
];

const DEFAULT_VISIBLE = 20;

function WordBucketRow({ bucket, tierClass }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`word-bucket-row ${tierClass}`}>
      <button
        type="button"
        className="word-bucket-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="word-bucket-word">{bucket.word}</span>
        <span className="word-bucket-count">{bucket.nlp_count} NLP{bucket.nlp_count === 1 ? '' : 's'}</span>
        <span className="word-bucket-toggle">{expanded ? '−' : '+'}</span>
      </button>
      {expanded && (
        <ul className="word-bucket-nlps">
          {bucket.nlps.map((nlp) => (
            <li key={nlp.text}>
              <span className="word-bucket-nlp-text">{nlp.text}</span>
              {nlp.combined_count > 0 && (
                <span className="word-bucket-nlp-count">{nlp.combined_count}x</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function TierSection({ tierKey, label, className, buckets }) {
  const [showAll, setShowAll] = useState(false);

  if (!buckets?.length) return null;

  const visibleBuckets = showAll ? buckets : buckets.slice(0, DEFAULT_VISIBLE);
  const hasMore = buckets.length > DEFAULT_VISIBLE;

  return (
    <section className={`word-buckets-tier ${className}`}>
      <div className="word-buckets-tier-header">
        <h4>{label} Word Buckets</h4>
        <span className="word-buckets-tier-meta">{buckets.length} words</span>
      </div>
      <div className="word-buckets-list">
        {visibleBuckets.map((bucket) => (
          <WordBucketRow key={bucket.word} bucket={bucket} tierClass={className} />
        ))}
      </div>
      {hasMore && (
        <button
          type="button"
          className="word-buckets-show-all"
          onClick={() => setShowAll((prev) => !prev)}
        >
          {showAll ? 'Show fewer' : `Show all (${buckets.length})`}
        </button>
      )}
    </section>
  );
}

function WordBucketsPanel({ mergeData }) {
  const wordBuckets = useMemo(() => resolveWordBuckets(mergeData || {}), [mergeData]);

  const hasBuckets = TIER_CONFIG.some(({ key }) => wordBuckets?.[key]?.length > 0);
  if (!hasBuckets) return null;

  return (
    <div className="word-buckets-panel">
      <div className="word-buckets-panel-header">
        <h3>Word Buckets by Tier</h3>
        <p>Words that appear in at least 2 distinct NLPs within each tier.</p>
      </div>
      {TIER_CONFIG.map(({ key, label, className }) => (
        <TierSection
          key={key}
          tierKey={key}
          label={label}
          className={className}
          buckets={wordBuckets[key]}
        />
      ))}
    </div>
  );
}

export default WordBucketsPanel;
