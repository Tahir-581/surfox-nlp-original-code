import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import apiClient from '../api/client';

function formatDate(value) {
  if (!value) return '—';
  return new Date(value).toLocaleString();
}

function HistoryPage({ onLoadSession }) {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState(null);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiClient.get('/searches');
      setItems(response.data.items || []);
    } catch (error) {
      toast.error('Could not load search history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const openSession = async (sessionId, viewMerge = false) => {
    try {
      const response = await apiClient.get(`/searches/${sessionId}`);
      const session = response.data;
      onLoadSession({
        sessionId: session.session_id,
        results: session.results || [],
        keyword: session.keyword || '',
        selectedUrls: session.selected_urls || [],
        mergeOutput: session.merge_output || null,
        timing: session.timing || {},
      });
      if (viewMerge && session.merge_output) {
        navigate('/merge', {
          state: {
            mergeData: session.merge_output,
            keyword: session.keyword,
            sessionId: session.session_id,
          },
        });
      } else {
        navigate(`/results/${session.session_id}`, {
          state: {
            results: session.results || [],
            sessionId: session.session_id,
            keyword: session.keyword || '',
            selectedUrls: session.selected_urls || [],
            mergeOutput: session.merge_output || null,
            timing: session.timing || {},
            searchTime: session.timing?.total_time_seconds || 0,
          },
        });
      }
    } catch (error) {
      toast.error('Could not load this search session');
    }
  };

  const handleDelete = async (sessionId) => {
    setDeletingId(sessionId);
    try {
      await apiClient.delete(`/searches/${sessionId}`);
      toast.success('Search deleted');
      setItems((prev) => prev.filter((item) => item.session_id !== sessionId));
    } catch (error) {
      toast.error('Could not delete search');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="page-container history-page">
      <div className="page-header">
        <h2>My Search History</h2>
        <p>Your saved searches and merged keyword results</p>
      </div>

      {loading ? (
        <div className="history-empty">Loading history...</div>
      ) : items.length === 0 ? (
        <div className="history-empty">
          <p>No searches yet.</p>
          <button className="btn-action btn-secondary" onClick={() => navigate('/')}>
            Run your first search
          </button>
        </div>
      ) : (
        <div className="history-table-container">
          <table className="results-table history-table">
            <thead>
              <tr>
                <th>Keyword</th>
                <th>Date</th>
                <th>Results</th>
                <th>Merge</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.session_id}>
                  <td>{item.keyword || item.session_id}</td>
                  <td>{formatDate(item.created_at)}</td>
                  <td>{item.result_count ?? 0}</td>
                  <td>{item.has_merge ? 'Yes' : 'No'}</td>
                  <td className="history-actions">
                    <button
                      className="btn-action btn-secondary"
                      onClick={() => openSession(item.session_id, false)}
                    >
                      View
                    </button>
                    {item.has_merge && (
                      <button
                        className="btn-action btn-secondary"
                        onClick={() => openSession(item.session_id, true)}
                      >
                        View merge
                      </button>
                    )}
                    <button
                      className="btn-action btn-danger"
                      disabled={deletingId === item.session_id}
                      onClick={() => handleDelete(item.session_id)}
                    >
                      {deletingId === item.session_id ? 'Deleting...' : 'Delete'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
