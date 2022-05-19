import React, { FC } from 'react';
import './SentenceWithTriggers.css';

interface SentenceWithTriggersProps {
  sentence: string;
  triggers: number[];
  triggerFrameIndices: { [trigger: number]: number };
}

const SentenceWithTriggers: FC<SentenceWithTriggersProps> = ({
  sentence,
  triggers,
  triggerFrameIndices,
}) => {
  return (
    <div className="SentenceWithTriggers">
      <div className="SentenceWithTriggers-mainSentence">{sentence}</div>
      {triggers.map(trigger => (
        <div className="SentenceWithTriggers-triggerContainer" key={trigger}>
          <span className="SentenceWithTriggers-triggerSentence">
            {sentence.slice(0, trigger + 1)}
            <span className="SentenceWithTriggers-triggerMarker">
              {triggerFrameIndices[trigger] !== undefined && (
                <span className="SentenceWithTriggers-frameLabel">
                  {triggerFrameIndices[trigger] + 1}
                </span>
              )}
            </span>
          </span>
        </div>
      ))}
    </div>
  );
};

export default SentenceWithTriggers;
