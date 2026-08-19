import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { config } from '../config';
import apiClient from '../api/client';
import { resolveVisibleTierEntities } from '../utils/finalNlpVisibility';
import KeywordInstancesPanel from '../components/KeywordInstancesPanel';
import GlinerLabelsPanel from '../components/GlinerLabelsPanel';

function EntityTooltip({ entity, rankingMethod = 'biencoder' }) {
  const method = rankingMethod === 'crossencoder' ? 'CrossEncoder' : 'BiEncoder';
  const competitorCount =
    entity.competitor_count ?? entity.found_in_files?.length ?? 0;

  return (
    <div className="entity-tooltip-live">
      <div className="tooltip-title-live">Term Details</div>
      <div className="tooltip-row-live">
        <span>Method:</span>
        <span>{method}</span>
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
      {entity.found_in_files?.length > 0 && (
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
            {entity.found_in_files.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}

function EntityCardLive({ entity, tier, showSources, rankingMethod }) {
  return (
    <div className={`entity-card-live ${tier}`}>
      <div className="entity-card-text-live">{entity.text}</div>
      <div className="entity-card-meta-live">
        <span className="entity-badge-count">{entity.combined_count}x</span>
        {showSources && visibleSourceBadges(entity.sources).length > 0 && (
          <span className="entity-sources">
            {visibleSourceBadges(entity.sources).map((src) => (
              <span key={src} className={`source-badge source-${src}`}>
                {src}
              </span>
            ))}
          </span>
        )}
      </div>
      <EntityTooltip entity={entity} rankingMethod={rankingMethod} />
    </div>
  );
}

const visibleSourceBadges = (sources) =>
  (Array.isArray(sources) ? sources : []).filter((src) => src !== 'gliner');

function ResultsPage({ sessionId: propSessionId, results: propResults, onLoadSession }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { sessionId: routeSessionId } = useParams();

  const [sessionId, setSessionId] = useState(
    routeSessionId || location.state?.sessionId || propSessionId || null
  );
  const [results, setResults] = useState(
    location.state?.results || propResults || []
  );
  const [selectedUrls, setSelectedUrls] = useState(() => {
    const fromState = location.state?.selectedUrls;
    if (fromState?.length) return fromState;
    if (location.state?.mergeOutput) {
      return (location.state?.results || propResults || [])
        .map((r) => r.url)
        .filter(Boolean);
    }
    return [];
  });
  const [mergeData, setMergeData] = useState(location.state?.mergeOutput || null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [loadingMerge, setLoadingMerge] = useState(false);
  const [keyword, setKeyword] = useState(location.state?.keyword || '');
  const [searchTime, setSearchTime] = useState(
    location.state?.searchTime || location.state?.timing?.total_time_seconds || 0
  );
  const [sourceFilter] = useState({ gpt: true, keybert: true, gliner: true });
  const [selectionDirty, setSelectionDirty] = useState(false);

  useEffect(() => {
    const activeSessionId = routeSessionId || location.state?.sessionId || propSessionId;
    if (activeSessionId) {
      setSessionId(activeSessionId);
      setSelectionDirty(false);
    }
  }, [routeSessionId, location.state?.sessionId, propSessionId]);

  useEffect(() => {
    const loadFromApi = async () => {
      const activeSessionId = routeSessionId || sessionId;
      if (!activeSessionId) return;

      const hasResults = (location.state?.results || propResults || results || []).length > 0;
      const needsFullSessionLoad = !hasResults;
      const needsMergeRefresh = Boolean(activeSessionId);

      if (!needsFullSessionLoad && !needsMergeRefresh) return;

      if (needsFullSessionLoad) {
        setLoadingSession(true);
      }
      try {
        const response = await apiClient.get(`/searches/${activeSessionId}`);
        const session = response.data;
        setSessionId(session.session_id);
        if (needsFullSessionLoad) {
          setResults(session.results || []);
          const sessionSelected = session.selected_urls || [];
          const sessionMerge = session.merge_output || null;
          setSelectedUrls(
            sessionSelected.length > 0
              ? sessionSelected
              : (sessionMerge ? (session.results || []).map((r) => r.url) : [])
          );
          setKeyword(session.keyword || '');
          setSearchTime(session.timing?.total_time_seconds || 0);
        }
        if (session.merge_output) {
          setMergeData(session.merge_output);
        }
        if (onLoadSession && needsFullSessionLoad) {
          onLoadSession({
            sessionId: session.session_id,
            results: session.results || [],
            keyword: session.keyword || '',
            selectedUrls: session.selected_urls || [],
            mergeOutput: session.merge_output || null,
            timing: session.timing || {},
          });
        }
      } catch (error) {
        console.error('Failed to load session:', error);
      } finally {
        if (needsFullSessionLoad) {
          setLoadingSession(false);
        }
      }
    };

    loadFromApi();
  }, [routeSessionId, sessionId, location.state?.results, propResults, onLoadSession, results.length]);

  const performRealTimeMerge = useCallback(async () => {
    if (!sessionId) return;
    setLoadingMerge(true);
    try {
      const response = await apiClient.post(config.MERGE_ENDPOINT, {
        selected_urls: selectedUrls,
        session_id: sessionId,
        keyword: keyword || undefined,
      });
      setMergeData(response.data);
    } catch (error) {
      console.error('Real-time merge error:', error);
    } finally {
      setLoadingMerge(false);
    }
  }, [selectedUrls, sessionId, keyword]);

  useEffect(() => {
    if (!selectionDirty) return;

    if (selectedUrls.length > 0) {
      performRealTimeMerge();
    } else {
      setMergeData(null);
    }
  }, [selectedUrls, selectionDirty, performRealTimeMerge]);

  const filteredEntities = (mergeData?.entities || []).filter((entity) => {
    const sources = Array.isArray(entity.sources) ? entity.sources : [];
    const hasGpt = sources.includes('gpt');
    const hasKeybert = sources.includes('keybert');
    const hasGliner = sources.includes('gliner');

    if (sources.length === 0) return sourceFilter.gpt || sourceFilter.keybert || sourceFilter.gliner;

    return (sourceFilter.gpt && hasGpt) || (sourceFilter.keybert && hasKeybert) || (sourceFilter.gliner && hasGliner);
  });

  const {
    highEntities,
    whiteEntities,
    mediumEntities,
    visibleFinalNlps,
    visibleStats,
  } = resolveVisibleTierEntities(mergeData, filteredEntities, keyword);

  if (loadingSession) {
    return (
      <div className="page-container results-page">
        <div className="page-header">
          <h2>Search Results</h2>
          <p>Loading session...</p>
        </div>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div className="page-container results-page">
        <div className="page-header">
          <h2>Search Results</h2>
          <p>No results available. Please search first or open a session from History.</p>
        </div>
        <button
          className="btn-action btn-secondary"
          onClick={() => navigate('/')}
        >
          Back to Search
        </button>
      </div>
    );
  }

  const handleSelectAll = (e) => {
    setSelectionDirty(true);
    if (e.target.checked) {
      setSelectedUrls(results.map((r) => r.url));
    } else {
      setSelectedUrls([]);
    }
  };

  const handleSelectUrl = (url) => {
    setSelectionDirty(true);
    setSelectedUrls((prev) => {
      if (prev.includes(url)) {
        return prev.filter((u) => u !== url);
      }
      return [...prev, url];
    });
  };

  const isAllSelected = results.length > 0 && selectedUrls.length === results.length;

  return (
    <div className="page-container results-page">
      <div className="page-header">
        <h2>Search Results</h2>
        <p>
          Select domains to analyze NLP terms in real-time
          {searchTime > 0 && <span style={{ color: '#0066cc', marginLeft: '1rem' }}>({Number(searchTime).toFixed(2)}s)</span>}
        </p>
        {keyword && (
          <p style={{ marginTop: '0.5rem', color: '#1e3a5f', fontWeight: 600 }}>
            Query: <span style={{ fontWeight: 700 }}>{keyword}</span>
          </p>
        )}
      </div>

      <div className="results-table-container">
        <div className="table-scroll-container">
          <table className="results-table">
            <thead>
              <tr>
                <th className="checkbox-col">
                  <input
                    type="checkbox"
                    checked={isAllSelected}
                    onChange={handleSelectAll}
                  />
                </th>
                <th className="rank-col">Rank</th>
                <th className="domain-col">Domain</th>
                <th>Title</th>
                <th>Words</th>
                <th>Authority</th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, idx) => (
                <tr key={idx}>
                  <td className="checkbox-col">
                    <input
                      type="checkbox"
                      checked={selectedUrls.includes(result.url)}
                      onChange={() => handleSelectUrl(result.url)}
                    />
                  </td>
                  <td className="rank-col">
                    <span className="word-count-badge">{result.rank ?? (idx + 1)}</span>
                  </td>
                  <td className="domain-col">
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="domain-cell"
                    >
                      {result.domain}
                    </a>
                  </td>
                  <td>
                    <div className="entity-preview">{result.title || 'N/A'}</div>
                  </td>
                  <td>
                    <span className="word-count-badge">{result.word_count}</span>
                  </td>
                  <td>
                    <span className="authority-badge">
                      {result.authority}/10
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {mergeData?.entities?.length > 0 && (
        <div className="merged-analysis-section">
          <div className="analysis-header">
            <h3>
              Live Analysis - {selectedUrls.length || mergeData.total_files_processed || 0} Domain(s)
            </h3>
            {loadingMerge && <span className="loading-small"></span>}
          </div>

          <div className="analysis-stats">
            <div className="stat-card-small">
              <div className="stat-label">Unique Entities</div>
              <div className="stat-value">{visibleStats.uniqueCount}</div>
            </div>
            <div className="stat-card-small">
              <div className="stat-label">Total Occurrences</div>
              <div className="stat-value">{visibleStats.occurrenceCount}</div>
            </div>
            <div className="stat-card-small">
              <div className="stat-label">Avg Words</div>
              <div className="stat-value">{mergeData.average_statistics?.avg_word_count || 0}</div>
            </div>
            <div className="stat-card-small">
              <div className="stat-label">Avg Headings</div>
              <div className="stat-value">{mergeData.average_statistics?.avg_heading_count || 0}</div>
            </div>
          </div>

          <div className="entities-header">
            <h3>Final NLP Terms</h3>
            <div className="nlp-filter-bar">
              <div className="nlp-filter-left">
                <p style={{ margin: 0 }}>
                  {visibleFinalNlps.length} terms (Hover for details)
                  {' '}·{' '}
                  <span style={{ color: '#059669', fontWeight: 600 }}>
                    Green: {highEntities.length}
                  </span>
                  {' '}·{' '}
                  <span style={{ color: '#ea580c', fontWeight: 600 }}>
                    Orange: {mediumEntities.length}
                  </span>
                  {' '}·{' '}
                  <span style={{ color: '#4b5563', fontWeight: 600 }}>
                    White: {whiteEntities.length}
                  </span>
                </p>
              </div>
            </div>
          </div>
          <div className="entities-grid-live">
            {highEntities.map((entity, idx) => (
              <EntityCardLive
                key={`high-${idx}`}
                entity={entity}
                tier="tier-high"
                showSources
                rankingMethod={mergeData.ranking_method || 'biencoder'}
              />
            ))}

            {mediumEntities.length > 0 && <div className="nlp-tier-break" />}
            {mediumEntities.map((entity, idx) => (
              <EntityCardLive
                key={`med-${idx}`}
                entity={entity}
                tier="tier-medium"
                rankingMethod={mergeData.ranking_method || 'biencoder'}
              />
            ))}

            {whiteEntities.length > 0 && <div className="nlp-tier-break" />}
            {whiteEntities.map((entity, idx) => (
              <EntityCardLive
                key={`white-${idx}`}
                entity={entity}
                tier="tier-low"
                rankingMethod={mergeData.ranking_method || 'biencoder'}
              />
            ))}
          </div>

          <GlinerLabelsPanel
            mergeData={mergeData}
            finalEntities={visibleFinalNlps}
          />

          <KeywordInstancesPanel mergeData={mergeData} visibleFinalNlps={visibleFinalNlps} />
        </div>
      )}

      <div className="action-bar">
        <div className="selection-info">
          {selectedUrls.length > 0 && (
            <span>
              Selected: <strong>{selectedUrls.length} of {results.length}</strong>
            </span>
          )}
        </div>
        <button
          className="btn-action btn-secondary"
          onClick={() => navigate('/')}
        >
          New Search
        </button>
        {mergeData && (
          <button
            className="btn-action btn-primary"
            onClick={() => navigate('/merge', { state: { mergeData, keyword, sessionId } })}
          >
            Open merge view
          </button>
        )}
      </div>
    </div>
  );
}

export default ResultsPage;
