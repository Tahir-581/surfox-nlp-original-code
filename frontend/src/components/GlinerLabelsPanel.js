import React, { useMemo, useState } from 'react';
import {
  buildTierLookup,
  countLabelsWithNlps,
  groupFinalNlpsByGlinerLabel,
  tierClassForEntity,
} from '../utils/glinerLabels';

function GlinerLabelNlpCard({ entity, tierClass, rankingMethod }) {
  const competitorCount =
    entity.competitor_count ?? entity.found_in_files?.length ?? 0;

  return (
    <div className={`entity-card-live ${tierClass}`}>
      <div className="entity-card-text-live">{entity.text}</div>
      <div className="entity-card-meta-live">
        <span className="entity-badge-count">{entity.combined_count ?? 0}x</span>
      </div>
      <div className="entity-tooltip-live">
        <div className="tooltip-title-live">Term Details</div>
        <div className="tooltip-row-live">
          <span>Method:</span>
          <span>{rankingMethod === 'crossencoder' ? 'CrossEncoder' : 'BiEncoder'}</span>
        </div>
        <div className="tooltip-row-live">
          <span>Score:</span>
          <span>
            {entity.average_weightage != null
              ? Number(entity.average_weightage).toFixed(3)
              : '—'}
          </span>
        </div>
        <div className="tooltip-row-live">
          <span>Count:</span>
          <span>{entity.combined_count ?? '—'}</span>
        </div>
        <div className="tooltip-row-live">
          <span>Found in:</span>
          <span>{competitorCount} domain(s)</span>
        </div>
      </div>
    </div>
  );
}

function GlinerLabelRow({
  label,
  nlps,
  defaultExpanded,
  tierLookup,
  rankingMethod,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className={`gliner-label-row ${nlps.length === 0 ? 'gliner-label-row-empty' : ''}`}>
      <button
        type="button"
        className="gliner-label-row-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="gliner-label-row-title">{label}</span>
        <span className="gliner-label-row-count">{nlps.length}</span>
        <span className="gliner-label-row-chevron">{expanded ? '▾' : '▸'}</span>
      </button>
      {expanded && (
        <div className="gliner-label-row-body">
          {nlps.length === 0 ? (
            <p className="gliner-label-empty">No NLPs</p>
          ) : (
            <div className="entities-grid-live gliner-label-nlp-grid">
              {nlps.map((entity, idx) => (
                <GlinerLabelNlpCard
                  key={`${label}-${entity.text}-${idx}`}
                  entity={entity}
                  tierClass={tierClassForEntity(entity, tierLookup)}
                  rankingMethod={rankingMethod}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function GlinerLabelsPanel({ mergeData, finalEntities }) {
  const grouped = useMemo(
    () =>
      groupFinalNlpsByGlinerLabel({
        glinerLabels: mergeData?.gliner_labels,
        entities: finalEntities,
        preGrouped: mergeData?.nlps_by_gliner_label,
        mergeData,
      }),
    [mergeData, finalEntities]
  );

  const tierLookup = useMemo(() => buildTierLookup(mergeData), [mergeData]);
  const visibleGrouped = useMemo(
    () => grouped.filter((row) => row.nlps.length > 0),
    [grouped]
  );
  const { populated, total } = useMemo(
    () => countLabelsWithNlps(visibleGrouped),
    [visibleGrouped]
  );

  if (!visibleGrouped.length) return null;

  return (
    <div className="gliner-labels-panel">
      <div className="gliner-labels-panel-header">
        <h3>GLiNER Labels</h3>
        <p>
          {populated} / {total} labels with NLPs · final terms grouped by GLiNER entity type
        </p>
      </div>
      <div className="gliner-labels-list">
        {visibleGrouped.map((row) => (
          <GlinerLabelRow
            key={row.label}
            label={row.label}
            nlps={row.nlps}
            defaultExpanded={row.nlps.length > 0}
            tierLookup={tierLookup}
            rankingMethod={mergeData?.ranking_method || 'biencoder'}
          />
        ))}
      </div>
    </div>
  );
}

export default GlinerLabelsPanel;
