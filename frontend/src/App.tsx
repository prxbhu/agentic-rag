import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Resources from './pages/Resources';
import Health from './pages/Health';

import { WorkspaceProvider } from './context/WorkspaceContext';

function App() {
  return (
    <WorkspaceProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="chat" element={<Chat />} />
            <Route path="chat/:conversationId" element={<Chat />} />
            <Route path="resources" element={<Resources />} />
            <Route path="health" element={<Health />} />
          </Route>
        </Routes>
      </Router>
    </WorkspaceProvider>
  );
}

export default App;