/* Runtime configuration - loaded by index.html */
(function () {
  // Empty API_URL = same-origin requests (/search, /merge, …).
  // Dev server proxies to backend; production build is served by backend on the same port.
  window.__APP_CONFIG__ = {
    API_URL: '',
    SEARCH_ENDPOINT: '/search',
    BATCH_SEARCH_ENDPOINT: '/batch_search',
    MERGE_ENDPOINT: '/merge',
  };
})();
