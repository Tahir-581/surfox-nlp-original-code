import React, { useCallback, useMemo, useState } from 'react';
import {
  countNlpsByTier,
  filterFinalNlpsForDisplay,
} from '../utils/finalNlpVisibility';

const TIER_CLASS = {
  green: 'tier-high',
  orange: 'tier-medium',
  white: 'tier-low',
};

const DEFAULT_VISIBLE_NLPS = 60;

function formatSim(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(3);
}

function groupNlpsByTier(nlps) {
  const green = [];
  const orange = [];
  const white = [];
  for (const nlp of nlps) {
    if (nlp.tier === 'green') green.push(nlp);
    else if (nlp.tier === 'white') white.push(nlp);
    else orange.push(nlp);
  }
  return { green, orange, white };
}

function InstanceNlpTooltip({ nlp, anchorText, domains }) {
  const domainList = domains?.length ? domains : [];
  const domainCount = nlp.competitor_count ?? domainList.length ?? 0;

  return (
    <div className="entity-tooltip-live">
      <div className="tooltip-title-live">Anchor Match</div>
      <div className="tooltip-row-live">
        <span>Anchor:</span>
        <span>{anchorText}</span>
      </div>
      <div className="tooltip-row-live">
        <span>Tier:</span>
        <span>{nlp.tier || '—'}</span>
      </div>
      <div className="tooltip-row-live">
        <span>Similarity:</span>
        <span>{formatSim(nlp.sim_to_query)}</span>
      </div>
      <div className="tooltip-row-live">
        <span>Count:</span>
        <span>{nlp.combined_count ?? '—'}</span>
      </div>
      {nlp.average_weightage != null && (
        <div className="tooltip-row-live">
          <span>Weightage:</span>
          <span>{Number(nlp.average_weightage).toFixed(3)}</span>
        </div>
      )}
      <div className="tooltip-row-live">
        <span>Found in:</span>
        <span>{domainCount} domain(s)</span>
      </div>
      {domainList.length > 0 && (
        <div
          style={{
            marginTop: '0.8rem',
            paddingTop: '0.8rem',
            borderTop: '1px solid rgba(255, 255, 255, 0.2)',
            fontSize: '0.75rem',
          }}
        >
          <div>Domains:</div>
          <div style={{ opacity: 0.9, marginTop: '0.3rem' }}>
            {domainList.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}

function InstanceNlpPill({ nlp, tierClass, anchorText, resolveDomains }) {
  const domains = resolveDomains(nlp);
  return (
    <div className={`entity-card-live ${tierClass}`}>
      <div className="entity-card-text-live">{nlp.text}</div>
      <div className="entity-card-meta-live">
        <span className="entity-badge-count">{nlp.combined_count ?? 0}x</span>
      </div>
      <InstanceNlpTooltip nlp={nlp} anchorText={anchorText} domains={domains} />
    </div>
  );
}

function InstanceNlpGrid({ nlps, anchorText, resolveDomains }) {
  const { green, orange, white } = groupNlpsByTier(nlps);

  return (
    <div className="entities-grid-live keyword-instance-pills">
      {green.map((nlp) => (
        <InstanceNlpPill
          key={`${anchorText}-g-${nlp.text}`}
          nlp={nlp}
          tierClass={TIER_CLASS.green}
          anchorText={anchorText}
          resolveDomains={resolveDomains}
        />
      ))}

      {orange.length > 0 && green.length > 0 && <div className="nlp-tier-break" />}
      {orange.map((nlp) => (
        <InstanceNlpPill
          key={`${anchorText}-o-${nlp.text}`}
          nlp={nlp}
          tierClass={TIER_CLASS.orange}
          anchorText={anchorText}
          resolveDomains={resolveDomains}
        />
      ))}

      {white.length > 0 && (green.length > 0 || orange.length > 0) && (
        <div className="nlp-tier-break" />
      )}
      {white.map((nlp) => (
        <InstanceNlpPill
          key={`${anchorText}-w-${nlp.text}`}
          nlp={nlp}
          tierClass={TIER_CLASS.white}
          anchorText={anchorText}
          resolveDomains={resolveDomains}
        />
      ))}
    </div>
  );
}

function InstanceRow({ instance, resolveDomains }) {
  const [expanded, setExpanded] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const roleLabel = instance.role === 'original' ? 'Original' : 'GLiNER';
  const nlps = useMemo(
    () => filterFinalNlpsForDisplay(instance.nlps || []),
    [instance.nlps]
  );
  const tierCounts = useMemo(() => countNlpsByTier(nlps), [nlps]);
  const visibleNlps = showAll ? nlps : nlps.slice(0, DEFAULT_VISIBLE_NLPS);
  const hasMore = nlps.length > DEFAULT_VISIBLE_NLPS;

  return (
    <div className={`keyword-instance-row role-${instance.role}`}>
      <button
        type="button"
        className="keyword-instance-header"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
      >
        <span className="keyword-instance-text">{instance.text}</span>
        <span className={`keyword-instance-role role-badge-${instance.role}`}>
          {roleLabel}
        </span>
        {instance.label && (
          <span className="keyword-instance-label">{instance.label}</span>
        )}
        {instance.gliner_score != null && (
          <span className="keyword-instance-score">
            {formatSim(instance.gliner_score)}
          </span>
        )}
        <span className="keyword-instance-count">
          {nlps.length} NLP{nlps.length === 1 ? '' : 's'}
        </span>
        <span className="keyword-instance-tier-summary">
          <span className="tier-summary-green">G: {tierCounts.green ?? 0}</span>
          {' · '}
          <span className="tier-summary-orange">O: {tierCounts.orange ?? 0}</span>
          {' · '}
          <span className="tier-summary-white">W: {tierCounts.white ?? 0}</span>
        </span>
        <span className="keyword-instance-toggle">{expanded ? '−' : '+'}</span>
      </button>
      {expanded && (
        <div className="keyword-instance-body">
          <InstanceNlpGrid
            nlps={visibleNlps}
            anchorText={instance.text}
            resolveDomains={resolveDomains}
          />
          {hasMore && (
            <button
              type="button"
              className="keyword-instances-show-all"
              onClick={() => setShowAll((prev) => !prev)}
            >
              {showAll ? 'Show fewer' : `Show all (${nlps.length})`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function KeywordInstancesPanel({ mergeData, visibleFinalNlps }) {
  const visibleTextKeys = useMemo(() => {
    const keys = new Set(
      (visibleFinalNlps || []).map((entity) => (entity.text || '').trim().toLowerCase())
    );
    if (!keys.size) return null;
    return keys;
  }, [visibleFinalNlps]);

  const instances = useMemo(() => {
    const raw = mergeData?.keyword_instances || [];
    return raw
      .map((instance) => {
        let nlps = filterFinalNlpsForDisplay(instance.nlps || []);
        if (visibleTextKeys) {
          nlps = nlps.filter((nlp) =>
            visibleTextKeys.has((nlp.text || '').trim().toLowerCase())
          );
        }
        return {
          ...instance,
          nlps,
          nlp_count: nlps.length,
          tier_counts: countNlpsByTier(nlps),
        };
      })
      .filter((instance) => instance.nlps.length > 0);
  }, [mergeData, visibleTextKeys]);

  const entityDomainMap = useMemo(() => {
    const map = new Map();
    const addEntity = (entity) => {
      const key = (entity?.text || '').trim().toLowerCase();
      if (!key || map.has(key)) return;
      const domains = entity.found_in_files;
      if (Array.isArray(domains) && domains.length > 0) {
        map.set(key, domains);
      }
    };
    for (const entity of mergeData?.entities || []) {
      addEntity(entity);
    }
    for (const tierKey of ['green_nlps', 'orange_nlps', 'white_nlps']) {
      for (const entity of mergeData?.[tierKey] || []) {
        addEntity(entity);
      }
    }
    return map;
  }, [mergeData]);

  const resolveDomains = useCallback(
    (nlp) => {
      if (nlp.found_in_files?.length) return nlp.found_in_files;
      return entityDomainMap.get((nlp.text || '').trim().toLowerCase()) || [];
    },
    [entityDomainMap]
  );

  if (!instances.length) return null;

  return (
    <div className="keyword-instances-panel">
      <div className="keyword-instances-panel-header">
        <h3>Keyword Instances</h3>
        <p>
          Per-anchor NLP tiers before the final proportional merge. Each instance is the
          original keyword or a GLiNER sub-entity; tiers are top 35% green, bottom 10%
          white, and middle orange by similarity to that anchor.
        </p>
      </div>
      <div className="keyword-instances-list">
        {instances.map((instance) => (
          <InstanceRow
            key={`${instance.role}-${instance.text}`}
            instance={instance}
            resolveDomains={resolveDomains}
          />
        ))}
      </div>
    </div>
  );
}

export default KeywordInstancesPanel;
