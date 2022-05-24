import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Intro.css';

const DEMO_SENTENCES = [
  'The hallway smelt of boiled cabbage and old rag mats.',
  'Prosperity has brought with it a new emphasis on historic preservation.',
  'The river forms a natural line between the north and south sections of the city.',
  'When the moon hits your eye, that\'s "amore".',
];

function Intro() {
  const [sentence, setSentence] = useState<string>('');
  const navigate = useNavigate();
  return (
    <div className="Intro">
      <header className="Intro-header">
        <h1>Frame Semantic Transformer Demo</h1>
        <p className="Intro-subtitle">
          Enter a sentence below to analyze the{' '}
          <a
            href="https://framenet2.icsi.berkeley.edu/"
            target="_blank"
            rel="noreferrer"
          >
            FrameNet
          </a>{' '}
          semantic frames in the text.
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
              maxLength={140}
            />
            <button
              className="Intro-querySubmit"
              disabled={sentence.trim() === ''}
            >
              Analyze
            </button>
          </div>
        </form>

        <div className="Intro-demoSentences">
          Need some inspiration? Try one of these:
          {DEMO_SENTENCES.map(sentence => (
            <div>
              <Link
                key={sentence}
                to={`/detect-frames?sentence=${encodeURIComponent(sentence)}`}
              >
                {sentence}
              </Link>
            </div>
          ))}
        </div>
      </header>
    </div>
  );
}

export default Intro;
