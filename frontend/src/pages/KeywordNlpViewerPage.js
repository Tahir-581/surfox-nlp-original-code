import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { toast } from 'react-toastify';
import apiClient from '../api/client';
import { filterSurferTerms, resolveSurferTierEntities } from '../utils/surferNlp';

function SurferEntityTooltip({ entity }) {
  const range = entity.target_range || {};
  return (
    <div className="entity-tooltip-live">
      <div className="tooltip-title-live">Surfer Term Details</div>
      <div className="tooltip-row-live">
        <span>Target range:</span>
        <span>
          {range.min ?? '—'} – {range.max ?? '—'}
        </span>
      </div>
      <div className="tooltip-row-live">
        <span>Midpoint:</span>
        <span>{entity.combined_count ?? '—'}</span>
      </div>
      <div className="tooltip-row-live">
        <span>Included:</span>
        <span>{entity.included ? 'Yes' : 'No'}</span>
      </div>
      <div className="tooltip-row-live">
        <span>NLP term:</span>
        <span>{entity.is_nlp ? 'Yes' : 'No'}</span>
      </div>
      <div className="tooltip-row-live">
        <span>Use in heading:</span>
        <span>{entity.use_in_heading ? 'Yes' : 'No'}</span>
      </div>
      {entity.ignored && (
        <div className="tooltip-row-live">
          <span>Ignored:</span>
          <span>Yes</span>
        </div>
      )}
    </div>
  );
}

function SurferEntityCard({ entity, tier }) {
  return (
    <div className={`entity-card-live ${tier}`}>
      <div className="entity-card-text-live">{entity.text}</div>
      <div className="entity-card-meta-live">
        <span className="entity-badge-count">{entity.combined_count}x</span>
      </div>
      <SurferEntityTooltip entity={entity} />
    </div>
  );
}

function KeywordNlpViewerPage() {
  const navigate = useNavigate();
  const { slug } = useParams();
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [includedOnly, setIncludedOnly] = useState(false);
  const [isNlpOnly, setIsNlpOnly] = useState(false);

  const loadPayload = useCallback(async () => {
    if (!slug) return;
    setLoading(true);
    try {
      const response = await apiClient.get(`/keyword-nlp/${slug}`);
      setPayload(response.data);
    } catch (error) {
      toast.error('Could not load keyword NLP output');
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, [slug]);

  useEffect(() => {
    loadPayload();
  }, [loadPayload]);

  const mergeView = payload?.merge_view;
  const filteredEntities = useMemo(
    () =>
      filterSurferTerms(mergeView?.entities || [], {
        includedOnly,
        isNlpOnly,
      }),
    [mergeView, includedOnly, isNlpOnly]
  );

  const { highEntities, whiteEntities } = useMemo(
    () => resolveSurferTierEntities(mergeView, filteredEntities),
    [mergeView, filteredEntities]
  );

  if (loading) {
    return (
      <div className="page-container results-page">
        <div className="page-header">
          <h2>Keyword NLP</h2>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className="page-container results-page">
        <div className="page-header">
          <h2>Keyword NLP</h2>
          <p>Could not load this keyword NLP output.</p>
        </div>
        <button
          type="button"
          className="btn-action btn-secondary"
          onClick={() => navigate('/keyword-nlp')}
        >
          Back to list
        </button>
      </div>
    );
  }

  return (
    <div className="page-container results-page">
      <div className="page-header">
        <h2>Keyword NLP</h2>
        <p>Surfer SEO NLP terms</p>
        <p style={{ marginTop: '0.5rem', color: '#1e3a5f', fontWeight: 600 }}>
          Query: <span style={{ fontWeight: 700 }}>{payload.keyword}</span>
        </p>
        {payload.surfer_link && (
          <p style={{ marginTop: '0.35rem' }}>
            <a href={payload.surfer_link} target="_blank" rel="noopener noreferrer">
              Open Surfer draft
            </a>
          </p>
        )}
      </div>

      <div className="merged-analysis-section">
        <div className="analysis-header">
          <h3>Surfer NLP Analysis</h3>
        </div>

        <div className="analysis-stats">
          <div className="stat-card-small">
            <div className="stat-label">Unique Terms</div>
            <div className="stat-value">{mergeView?.total_unique_entities ?? 0}</div>
          </div>
          <div className="stat-card-small">
            <div className="stat-label">Target midpoint sum</div>
            <div className="stat-value">{mergeView?.total_entity_occurrences ?? 0}</div>
          </div>
          <div className="stat-card-small">
            <div className="stat-label">Included</div>
            <div className="stat-value">{mergeView?.green_nlps?.length ?? 0}</div>
          </div>
          <div className="stat-card-small">
            <div className="stat-label">Not included</div>
            <div className="stat-value">{mergeView?.white_nlps?.length ?? 0}</div>
          </div>
        </div>

        <div className="entities-header">
          <h3>Final NLP Terms</h3>
          <div className="nlp-filter-bar">
            <div className="nlp-filter-left">
              <p style={{ margin: 0 }}>
                {filteredEntities.length} terms (Hover for details)
                {' '}·{' '}
                <span style={{ color: '#059669', fontWeight: 600 }}>
                  Green: {highEntities.length}
                </span>
                {' '}·{' '}
                <span style={{ color: '#4b5563', fontWeight: 600 }}>
                  White: {whiteEntities.length}
                </span>
              </p>
            </div>
            <div className="nlp-filter-right" style={{ display: 'flex', gap: '1rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <input
                  type="checkbox"
                  checked={includedOnly}
                  onChange={(e) => setIncludedOnly(e.target.checked)}
                />
                Included only
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <input
                  type="checkbox"
                  checked={isNlpOnly}
                  onChange={(e) => setIsNlpOnly(e.target.checked)}
                />
                NLP terms only
              </label>
            </div>
          </div>
        </div>

        <div className="entities-grid-live">
          {highEntities.map((entity, idx) => (
            <SurferEntityCard key={`green-${entity.text}-${idx}`} entity={entity} tier="tier-high" />
          ))}

          {whiteEntities.length > 0 && <div className="nlp-tier-break" />}
          {whiteEntities.map((entity, idx) => (
            <SurferEntityCard key={`white-${entity.text}-${idx}`} entity={entity} tier="tier-low" />
          ))}
        </div>
      </div>

      <div className="action-bar">
        <button
          type="button"
          className="btn-action btn-secondary"
          onClick={() => navigate('/keyword-nlp')}
        >
          Back to list
        </button>
      </div>
    </div>
  );
}

export default KeywordNlpViewerPage;
