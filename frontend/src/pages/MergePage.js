import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import apiClient from '../api/client';
import { resolveVisibleTierEntities } from '../utils/finalNlpVisibility';
import WordBucketsPanel from '../components/WordBucketsPanel';

function MergePage({ sessionId: propSessionId }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { sessionId: routeSessionId } = useParams();
  const [mergeData, setMergeData] = useState(location.state?.mergeData || null);
  const [loading, setLoading] = useState(false);
  const [keyword, setKeyword] = useState(location.state?.keyword || '');

  const sessionId = location.state?.sessionId || routeSessionId || propSessionId;

  useEffect(() => {
    const loadMerge = async () => {
      if (mergeData || !sessionId) return;
      setLoading(true);
      try {
        const response = await apiClient.get(`/searches/${sessionId}`);
        const session = response.data;
        if (session.merge_output) {
          setMergeData(session.merge_output);
        }
        setKeyword(session.keyword || '');
      } catch (error) {
        console.error('Failed to load merge data:', error);
      } finally {
        setLoading(false);
      }
    };
    loadMerge();
  }, [mergeData, sessionId]);

  if (loading) {
    return (
      <div className="page-container merge-page">
        <div className="page-header">
          <h2>Analysis Results</h2>
          <p>Loading merge data...</p>
        </div>
      </div>
    );
  }

  if (!mergeData) {
    return (
      <div className="page-container merge-page">
        <div className="page-header">
          <h2>Analysis Results</h2>
          <p>No merge data available. Select URLs on the Results page first.</p>
        </div>
        <button
          className="btn-action btn-secondary"
          onClick={() => navigate(sessionId ? `/results/${sessionId}` : '/')}
        >
          Back
        </button>
      </div>
    );
  }

  const stats = mergeData.average_statistics || {};
  const entities = mergeData.entities || [];

  const {
    highEntities,
    whiteEntities,
    mediumEntities,
    visibleStats,
  } = resolveVisibleTierEntities(mergeData, entities, keyword);

  return (
    <div className="page-container merge-page">
      <div className="page-header">
        <h2>Entity Analysis</h2>
        <p>Aggregated insights from {mergeData.total_files_processed} domains</p>
      </div>

      <div className="merge-stats">
        <div className="stat-card">
          <h3>Unique Entities</h3>
          <div className="value">{visibleStats.uniqueCount}</div>
        </div>
        <div className="stat-card">
          <h3>Total Occurrences</h3>
          <div className="value">{visibleStats.occurrenceCount}</div>
        </div>
        <div className="stat-card">
          <h3>Avg Word Count</h3>
          <div className="value">{stats.avg_word_count || 0}</div>
        </div>
        <div className="stat-card">
          <h3>Avg Headings</h3>
          <div className="value">{stats.avg_heading_count || 0}</div>
        </div>
      </div>

      <div className="entities-grid">
        {highEntities.map((entity, idx) => (
          <div key={`high-${idx}`} className="entity-card tier-high">
            <div className="entity-text">{entity.text}</div>
            <div className="entity-meta">
              <span>{entity.combined_count}x</span>
              <span>{entity.competitor_count} domains</span>
            </div>
          </div>
        ))}
        {mediumEntities.map((entity, idx) => (
          <div key={`med-${idx}`} className="entity-card tier-medium">
            <div className="entity-text">{entity.text}</div>
            <div className="entity-meta">
              <span>{entity.combined_count}x</span>
            </div>
          </div>
        ))}
        {whiteEntities.map((entity, idx) => (
          <div key={`white-${idx}`} className="entity-card tier-low">
            <div className="entity-text">{entity.text}</div>
            <div className="entity-meta">
              <span>{entity.combined_count}x</span>
            </div>
          </div>
        ))}
      </div>

      <WordBucketsPanel mergeData={mergeData} />

      <div className="action-bar">
        <button
          className="btn-action btn-secondary"
          onClick={() => navigate(sessionId ? `/results/${sessionId}` : '/history')}
        >
          Back to results
        </button>
      </div>
    </div>
  );
}

export default MergePage;
