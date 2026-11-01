import React, { useState, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import SearchPage from './pages/SearchPage';
import ResultsPage from './pages/ResultsPage';
import MergePage from './pages/MergePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import HistoryPage from './pages/HistoryPage';
import KeywordNlpListPage from './pages/KeywordNlpListPage';
import KeywordNlpViewerPage from './pages/KeywordNlpViewerPage';
import Header from './components/Header';

function AppRoutes() {
  const [sessionId, setSessionId] = useState(null);
  const [searchResults, setSearchResults] = useState([]);

  const handleSearchComplete = useCallback((data) => {
    setSessionId(data.session_id);
    setSearchResults(data.results);
  }, []);

  const handleLoadSession = useCallback((data) => {
    setSessionId(data.sessionId);
    setSearchResults(data.results || []);
  }, []);

  return (
    <div className="app">
      <Header />
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={true}
      />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <SearchPage onSearchComplete={handleSearchComplete} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <HistoryPage onLoadSession={handleLoadSession} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/results/:sessionId?"
          element={
            <ProtectedRoute>
              <ResultsPage
                sessionId={sessionId}
                results={searchResults}
                onLoadSession={handleLoadSession}
              />
            </ProtectedRoute>
          }
        />
        <Route
          path="/keyword-nlp"
          element={
            <ProtectedRoute>
              <KeywordNlpListPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/keyword-nlp/:slug"
          element={
            <ProtectedRoute>
              <KeywordNlpViewerPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/merge"
          element={
            <ProtectedRoute>
              <MergePage sessionId={sessionId} />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </Router>
  );
}

export default App;
