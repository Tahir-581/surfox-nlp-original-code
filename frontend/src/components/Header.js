import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function Header() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="header">
      <div className="header-content">
        <h1>Surfox</h1>
        <nav className="header-nav">
          {isAuthenticated ? (
            <>
              <Link to="/" className="nav-link">Home</Link>
              <Link to="/history" className="nav-link">History</Link>
              <Link to="/keyword-nlp" className="nav-link">Keyword NLP</Link>
              <span className="nav-user">{user?.name || user?.email}</span>
              <button type="button" className="nav-link nav-button" onClick={handleLogout}>
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login" className="nav-link">Login</Link>
              <Link to="/register" className="nav-link">Register</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

export default Header;
