// Configuration loader that supports both development and production
// This is a LAZY getter to ensure window.__APP_CONFIG__ is available when accessed
export const config = {
  get API_URL() {
    // Priority 1: Check if runtime config was loaded
    if (typeof window !== 'undefined' && window.__APP_CONFIG__ && window.__APP_CONFIG__.API_URL) {
      return window.__APP_CONFIG__.API_URL;
    }
    
    // Priority 2: Environment variables from build time
    if (process.env.REACT_APP_API_URL) {
      return process.env.REACT_APP_API_URL;
    }
    
    // Priority 3: Same-origin (dev proxy or backend-served build)
    return '';
  },
  
  get SEARCH_ENDPOINT() {
    if (typeof window !== 'undefined' && window.__APP_CONFIG__ && window.__APP_CONFIG__.SEARCH_ENDPOINT) {
      return window.__APP_CONFIG__.SEARCH_ENDPOINT;
    }
    return process.env.REACT_APP_SEARCH_ENDPOINT || '/search';
  },

  get BATCH_SEARCH_ENDPOINT() {
    if (typeof window !== 'undefined' && window.__APP_CONFIG__ && window.__APP_CONFIG__.BATCH_SEARCH_ENDPOINT) {
      return window.__APP_CONFIG__.BATCH_SEARCH_ENDPOINT;
    }
    return process.env.REACT_APP_BATCH_SEARCH_ENDPOINT || '/batch_search';
  },
  
  get MERGE_ENDPOINT() {
    if (typeof window !== 'undefined' && window.__APP_CONFIG__ && window.__APP_CONFIG__.MERGE_ENDPOINT) {
      return window.__APP_CONFIG__.MERGE_ENDPOINT;
    }
    return process.env.REACT_APP_MERGE_ENDPOINT || '/merge';
  }
};
