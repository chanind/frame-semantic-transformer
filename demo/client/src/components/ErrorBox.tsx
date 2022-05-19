import React, { FC } from 'react';
import './ErrorBox.css';

interface ErrorBoxProps {
  message: string;
}

const ErrorBox: FC<ErrorBoxProps> = ({ message }) => {
  return <div className="ErrorBox">{message}</div>;
};

export default ErrorBox;
