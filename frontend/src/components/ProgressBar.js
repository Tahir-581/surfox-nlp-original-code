import React from 'react';
import '../styles/ProgressBar.css';

function ProgressBar({ progress }) {
  return (
    <div className="progress-bar-container">
      <div className="progress-bar-fill" style={{ width: `${progress}%` }}>
        <span className="progress-text">{Math.round(progress)}%</span>
      </div>
    </div>
  );
}

export default ProgressBar;
