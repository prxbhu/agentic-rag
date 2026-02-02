import { useState, useEffect } from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { MessageSquare, FileText, Activity, Moon, Sun, MessageCircle, ChevronDown } from 'lucide-react';
import { conversationApi, Conversation } from '@/lib/api';
import { useWorkspace } from '@/context/WorkspaceContext';
import WorkspaceSwitcher from './WorkspaceSwitcher';

export default function Layout() {
  const { currentWorkspace } = useWorkspace();
  const navigate = useNavigate();
  const location = useLocation();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark' || 
           (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  // Reset pagination when workspace changes
  useEffect(() => {
    setConversations([]);
    setPage(0);
    setHasMore(true);
    if (currentWorkspace) {
      loadConversations(0, true);
    }
  }, [currentWorkspace]);

  // Also refresh (but keep list) if we navigate to a new chat (to see new title)
  // This is a simple approach; ideal is to re-fetch or prepend.
  useEffect(() => {
    if (currentWorkspace && location.pathname.startsWith('/chat')) {
        // If we are on chat, maybe just reload the first page to get recent updates
        loadConversations(0, true);
    }
  }, [location.pathname]);

  const loadConversations = async (pageNum: number, reset = false) => {
    if (!currentWorkspace || loadingHistory) return;
    setLoadingHistory(true);
    try {
      const response = await conversationApi.list(currentWorkspace.id, 20, pageNum * 20);
      const newConvs = response.data.conversations || [];
      
      if (newConvs.length < 20) {
        setHasMore(false);
      } else {
        setHasMore(true);
      }

      if (reset) {
        setConversations(newConvs);
        setPage(1);
      } else {
        setConversations(prev => [...prev, ...newConvs]);
        setPage(pageNum + 1);
      }
    } catch (err) {
      console.error("Failed to load conversations", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  const navItems = [
    { to: '/chat', icon: MessageSquare, label: 'New Chat' },
    { to: '/resources', icon: FileText, label: 'Resources' },
    { to: '/health', icon: Activity, label: 'Health' },
  ];

  return (
    <div className="h-screen flex bg-gray-50 dark:bg-gray-900 transition-colors duration-200 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col flex-shrink-0 h-full">
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-primary-600 dark:text-primary-400 px-2">RAG System</h1>
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-300"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
          <WorkspaceSwitcher />
        </div>

        <nav className="flex-none p-4 space-y-2">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/chat'} // Exact match for new chat
              className={({ isActive }) =>
                `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${isActive
                  ? 'bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300 font-medium'
                  : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`
              }
            >
              <Icon size={20} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Conversation History - Scrollable */}
        <div className="flex-1 overflow-y-auto px-4 pb-4 min-h-0">
          <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3 px-2 sticky top-0 bg-white dark:bg-gray-800 py-2 z-10">
            History
          </div>
          <div className="space-y-1">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => navigate(`/chat/${conv.id}`)}
                className={`w-full text-left flex items-center space-x-3 px-3 py-2 rounded-md transition-colors text-sm truncate ${
                  location.pathname === `/chat/${conv.id}`
                    ? 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                <MessageCircle size={14} className="flex-shrink-0" />
                <span className="truncate">{conv.title || 'Untitled'}</span>
              </button>
            ))}
            {conversations.length === 0 && !loadingHistory && (
              <p className="text-xs text-gray-400 dark:text-gray-500 px-2 italic">No history yet</p>
            )}
            
            {hasMore && (
                <button 
                    onClick={() => loadConversations(page)}
                    disabled={loadingHistory}
                    className="w-full text-center text-xs text-primary-600 dark:text-primary-400 py-2 hover:underline disabled:opacity-50"
                >
                    {loadingHistory ? 'Loading...' : 'Load more'}
                </button>
            )}
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 flex-shrink-0">
          <p>© 2026 RAG System</p>
          <p className="mt-1">Version 1.0.0</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-gray-50 dark:bg-gray-900 w-full">
        <Outlet />
      </main>
    </div>
  );
}