import React, { FC } from 'react';
import './Frame.css';
import { ExternalLinkIcon } from '@heroicons/react/solid';

interface FrameProps {
  name: string;
  index: number;
  elements: {
    name: string;
    text: string;
  }[];
}

const Frame: FC<FrameProps> = ({ name, index, elements }) => {
  return (
    <div className="Frame">
      <div className="Frame-number">{index + 1}</div>
      <div className="Frame-name">
        {name}
        <a
          href={`https://framenet2.icsi.berkeley.edu/fnReports/data/frame/${name}.xml`}
          target="_blank"
          rel="noreferrer"
        >
          <ExternalLinkIcon className="Frame-extLinkIcon" />
        </a>
      </div>
      <div className="Frame-elements">
        {elements.map((element, i) => (
          <div key={i} className="Frame-element">
            <div className="Frame-elementName">{element.name}</div>
            <div className="Frame-elementText">{element.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Frame;
