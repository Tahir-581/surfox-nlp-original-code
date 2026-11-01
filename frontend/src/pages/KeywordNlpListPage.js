import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import apiClient from '../api/client';

function KeywordNlpListPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadItems = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiClient.get('/keyword-nlp');
      setItems(response.data.items || []);
    } catch (error) {
      toast.error('Could not load keyword NLP outputs');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadItems();
  }, [loadItems]);

  return (
    <div className="page-container history-page">
      <div className="page-header">
        <h2>Keyword NLP</h2>
        <p>Surfer SEO NLP exports from keyword-nlp-output</p>
      </div>

      {loading ? (
        <div className="history-empty">Loading keyword NLP outputs...</div>
      ) : items.length === 0 ? (
        <div className="history-empty">
          <p>No keyword NLP files found.</p>
        </div>
      ) : (
        <div className="history-table-container">
          <table className="results-table history-table">
            <thead>
              <tr>
                <th>Keyword</th>
                <th>Status</th>
                <th>Surfer draft</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr
                  key={item.slug}
                  className="keyword-nlp-row"
                  onClick={() => navigate(`/keyword-nlp/${item.slug}`)}
                  style={{ cursor: 'pointer' }}
                >
                  <td>{item.keyword}</td>
                  <td>{item.status || '—'}</td>
                  <td>
                    {item.surfer_link ? (
                      <a
                        href={item.surfer_link}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Open in Surfer
                      </a>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="history-actions">
                    <button
                      type="button"
                      className="btn-action btn-secondary"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/keyword-nlp/${item.slug}`);
                      }}
                    >
                      View NLPs
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

export default KeywordNlpListPage;
