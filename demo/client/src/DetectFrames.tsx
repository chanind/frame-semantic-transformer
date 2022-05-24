import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ClapSpinner } from 'react-spinners-kit';
import ErrorBox from './components/ErrorBox';
import Frame from './components/Frame';
import SentenceWithTriggers from './components/SentenceWithTriggers';
import { API_HOST } from './config';
import './DetectFrames.css';

const useUrlParams = () => {
  return new URLSearchParams(useLocation().search);
};

interface DetectFramesResponse {
  sentence: string;
  trigger_locations: number[];
  frames: {
    name: string;
    trigger_location: number;
    frame_elements: {
      name: string;
      text: string;
    }[];
  }[];
}

function DetectFrames() {
  const urlParams = useUrlParams();
  const sentenceQuery = urlParams.get('sentence') || '';
  const [sentence, setSentence] = useState(sentenceQuery);
  const { isLoading, error, data } = useQuery<DetectFramesResponse, any>(
    `detect-frames/${sentenceQuery}`,
    () =>
      fetch(
        API_HOST +
          '/detect-frames?sentence=' +
          encodeURIComponent(sentenceQuery),
      ).then(res => res.json()),
  );
  const triggerFrameIndices: { [trigger: number]: number } = {};
  data?.frames.forEach((frame, i) => {
    triggerFrameIndices[frame.trigger_location] = i;
  });
  const navigate = useNavigate();
  return (
    <div className="DetectFrames">
      <header className="DetectFrames-header">
        <Link className="DetectFrames-titleLink" to="/">
          <h1>Frame Semantic Transformer Demo</h1>
        </Link>

        <form
          className="DetectFrames-searchForm"
          onSubmit={evt => {
            evt.preventDefault();
            if (sentence.trim() !== '') {
              navigate(
                `/detect-frames?sentence=${encodeURIComponent(sentence)}`,
              );
            }
          }}
        >
          <div className="DetectFrames-query">
            <input
              className="DetectFrames-queryInput"
              type="text"
              placeholder="Enter a sentence"
              value={sentence}
              onChange={evt => setSentence(evt.target.value)}
              maxLength={140}
            />
            <button
              className="DetectFrames-querySubmit"
              disabled={sentence.trim() === ''}
            >
              Update
            </button>
          </div>
        </form>
      </header>
      {isLoading && (
        <div className="DetectFrames-loading">
          <ClapSpinner size={40} frontColor="#61dafb" loading={isLoading} />
          <p className="DetectFrames-loadingNotice">
            May take a few minutes on first load
          </p>
        </div>
      )}
      {data && (
        <>
          <div className="DetectFrames-sentenceWithTriggers">
            <SentenceWithTriggers
              sentence={data.sentence}
              triggers={data.trigger_locations}
              triggerFrameIndices={triggerFrameIndices}
            />
          </div>
          <div className="DetectFrames-analysis">
            {data.frames.map((frame, i) => (
              <div className="DetectFrames-frame" key={i}>
                <Frame
                  name={frame.name}
                  index={i}
                  elements={frame.frame_elements}
                />
              </div>
            ))}
          </div>
        </>
      )}
      {error && (
        <div>
          <ErrorBox message={`An error occurred: ${error?.message}`} />
        </div>
      )}
    </div>
  );
}

export default DetectFrames;
