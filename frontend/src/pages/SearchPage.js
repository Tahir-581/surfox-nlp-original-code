import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import { config } from '../config';
import apiClient from '../api/client';

function SearchPage({ onSearchComplete }) {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [batchMode, setBatchMode] = useState(false);
  const [batchKeywords, setBatchKeywords] = useState('');
  const [formData, setFormData] = useState({
    keyword: '',
    k: 20,
    device: 'mobile',
    use_proxy: true,
    use_browser: true
  });

  // Timer effect
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => {
        setElapsed(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseInt(value) : value)
    }));
  };

  const handleSearch = async (e) => {
    e.preventDefault();

    const parseBatchKeywords = (raw) => {
      const txt = (raw || '').trim();
      if (!txt) return [];

      // Preferred input: JSON array of strings (as provided by user)
      if (txt.startsWith('[')) {
        try {
          const arr = JSON.parse(txt);
          if (Array.isArray(arr)) {
            return arr
              .map(v => (typeof v === 'string' ? v.trim() : ''))
              .filter(Boolean)
              .slice(0, 50);
          }
        } catch (err) {
          // fall back to newline parsing below
        }
      }

      // Fallback: one per line
      return txt
        .split('\n')
        .map(s => s.trim())
        .filter(Boolean)
        .slice(0, 50);
    };

    const parsedBatch = parseBatchKeywords(batchKeywords);

    if (batchMode) {
      if (parsedBatch.length < 2) {
        toast.error('Please enter at least 2 titles/keywords (JSON array or one per line)');
        return;
      }
    } else {
      if (!formData.keyword.trim()) {
        toast.error('Please enter a keyword');
        return;
      }
    }

    setLoading(true);
    setProgress(10);
    setElapsed(0);
    const startTime = Date.now();

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + Math.random() * 15, 95));
      }, 800);

      const endpoint = batchMode ? config.BATCH_SEARCH_ENDPOINT : config.SEARCH_ENDPOINT;

      const response = await apiClient.post(
        endpoint,
        batchMode
          ? {
              keywords: parsedBatch,
              k: formData.k,
              device: formData.device,
              use_proxy: formData.use_proxy,
              headless: !formData.use_browser,
              use_browser: formData.use_browser
            }
          : {
              keyword: formData.keyword,
              k: formData.k,
              device: formData.device,
              use_proxy: formData.use_proxy,
              headless: !formData.use_browser,
              use_browser: formData.use_browser
            },
        { timeout: 600000 }
      );

      clearInterval(progressInterval);
      setProgress(100);

      // Calculate search time in seconds
      const endTime = Date.now();
      const searchTime = (endTime - startTime) / 1000;

      setTimeout(() => {
        if (batchMode) {
          toast.success(`Batch complete: ${response.data.completed}/${response.data.total_keywords} saved`);
          const firstOk = (response.data.items || []).find(i => i && !i.error);
          if (firstOk && firstOk.session_id) {
            navigate('/results', {
              state: {
                results: [],
                sessionId: firstOk.session_id,
                keyword: firstOk.keyword,
                searchTime: searchTime,
                timing: firstOk.timing || {},
                batchSummary: response.data
              }
            });
          }
        } else {
          onSearchComplete(response.data);
          navigate(`/results/${response.data.session_id}`, { 
            state: { 
              results: response.data.results,
              sessionId: response.data.session_id,
              keyword: formData.keyword,
              searchTime: searchTime,
              timing: response.data.timing,
              mergeOutput: response.data.merge_output || null,
              selectedUrls: response.data.selected_urls || [],
            }
          });
        }
      }, 500);

    } catch (error) {
      console.error('Search error:', error);
      const apiUrl = config.API_URL;
      const message =
        error.response?.data?.detail ||
        (error.request
          ? `Cannot reach backend at ${apiUrl}. Make sure backend is running and API URL/port is correct.`
          : error.message) ||
        'Search failed. Please try again.';
      toast.error(message);
    } finally {
      setLoading(false);
      setProgress(0);
      setElapsed(0);
    }
  };

  return (
    <div className="page-container search-page">
      <div className={`search-card ${batchMode ? 'search-card-wide' : ''}`}>
        <h2>Domain Analysis</h2>
        <p>Search Google and extract SEO-friendly NLP terms from competitor domains</p>

        <form onSubmit={handleSearch}>
          {/* Batch Mode Toggle */}
          <div className="checkbox-group" style={{ marginBottom: 12 }}>
            <div className="checkbox-item">
              <input
                id="batch_mode"
                type="checkbox"
                checked={batchMode}
                onChange={(e) => setBatchMode(e.target.checked)}
                disabled={loading}
              />
              <label htmlFor="batch_mode">Batch mode (multiple titles/keywords)</label>
            </div>
          </div>

          {/* Keyword Input */}
          <div className="form-group">
            <label htmlFor="keyword">Search Keyword *</label>
            <input
              id="keyword"
              type="text"
              name="keyword"
              placeholder="e.g., best dog breeds for families"
              value={formData.keyword}
              onChange={handleInputChange}
              disabled={loading || batchMode}
            />
          </div>

          {/* Batch Keywords Input */}
          {batchMode && (
            <div className="form-group">
              <label htmlFor="batch_keywords">Titles / Keywords (one per line)</label>
              <textarea
                id="batch_keywords"
                placeholder={'Paste either:\n1) JSON array: ["Title 1", "Title 2"]\n2) One per line:\nTitle 1\nTitle 2'}
                value={batchKeywords}
                onChange={(e) => setBatchKeywords(e.target.value)}
                disabled={loading}
                rows={14}
                style={{ resize: 'vertical', minHeight: 260 }}
              />
            </div>
          )}

          {/* Results Count */}
          <div className="form-group">
            <label htmlFor="k">Number of Results</label>
            <input
              id="k"
              type="number"
              name="k"
              min="5"
              max="50"
              value={formData.k}
              onChange={handleInputChange}
              disabled={loading}
            />
          </div>

          {/* Device Type */}
          <div className="form-group">
            <label htmlFor="device">Device Type</label>
            <select
              id="device"
              name="device"
              value={formData.device}
              onChange={handleInputChange}
              disabled={loading}
            >
              <option value="desktop">Desktop</option>
              <option value="mobile">Mobile</option>
            </select>
          </div>

          {/* Checkboxes */}
          <div className="checkbox-group">
            <div className="checkbox-item">
              <input
                id="browser"
                type="checkbox"
                name="use_browser"
                checked={formData.use_browser}
                onChange={handleInputChange}
                disabled={loading}
              />
              <label htmlFor="browser">Use Browser (visible window)</label>
            </div>
            <div className="checkbox-item">
              <input
                id="usa_proxy"
                type="checkbox"
                name="use_proxy"
                checked={formData.use_proxy}
                onChange={handleInputChange}
                disabled={loading}
              />
              <label htmlFor="usa_proxy">🇺🇸 USA Location</label>
            </div>
          </div>

          {/* Submit Button with Inline Timer */}
          <div className="button-timer-wrapper">
            <button 
              type="submit" 
              className="btn-search"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="loading"></span>
                  Searching...
                </>
              ) : (
                'Search Google'
              )}
            </button>
            
            {loading && (
              <div className="inline-timer">
                <span className="timer-icon">⏱️</span>
                <span className="timer-value">{formatTime(elapsed)}</span>
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {loading && progress > 0 && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${progress}%` }}></div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default SearchPage;
