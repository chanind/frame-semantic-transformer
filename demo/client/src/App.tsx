import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';

import './App.css';
import OctocatCorner from './components/OctocatCorner';
import DetectFrames from './DetectFrames';
import Intro from './Intro';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <OctocatCorner />
        <Router>
          <Routes>
            <Route path="/" element={<Intro />} />
            <Route path="/detect-frames" element={<DetectFrames />} />
          </Routes>
        </Router>
      </div>
    </QueryClientProvider>
  );
}

export default App;
