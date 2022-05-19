import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Intro.css';

function Intro() {
  const [sentence, setSentence] = useState<string>('');
  const navigate = useNavigate();
  return (
    <div className="Intro">
      <header className="Intro-header">
        <h1>Frame Semantic Transformer Demo</h1>
        <p className="Intro-subtitle">
          Enter a sentence below to analyze the{' '}
          <a href="https://framenet2.icsi.berkeley.edu/">FrameNet</a> semantic
          frames in the text.
        </p>
        <form
          className="Intro-searchForm"
          onSubmit={evt => {
            evt.preventDefault();
            if (sentence.trim() !== '') {
              navigate(
                `/detect-frames?sentence=${encodeURIComponent(sentence)}`,
              );
            }
          }}
        >
          <div className="Intro-query">
            <input
              className="Intro-queryInput"
              type="text"
              placeholder="Enter a sentence"
              value={sentence}
              onChange={evt => setSentence(evt.target.value)}
            />
            <button
              className="Intro-querySubmit"
              disabled={sentence.trim() === ''}
            >
              Analyze
            </button>
          </div>
        </form>
      </header>
    </div>
  );
}

export default Intro;
