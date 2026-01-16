import { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { Send, Loader2, FileText, Plus } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { conversationApi, Message } from '@/lib/api';
import { format } from 'date-fns';
import { useWorkspace } from '@/context/WorkspaceContext';

export default function Chat() {
  const { currentWorkspace } = useWorkspace();
  const { conversationId: urlConversationId } = useParams();
  const [conversationId, setConversationId] = useState<string | null>(
    urlConversationId || null
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (conversationId) {
      loadConversation();
    }
  }, [conversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversation = async () => {
    if (!conversationId) return;
    try {
      const response = await conversationApi.get(conversationId);
      setMessages(response.data.messages || []);
    } catch (err) {
      setError('Failed to load conversation');
      console.error(err);
    }
  };

  const createNewConversation = async () => {
    if (!currentWorkspace) {
      setError('Please select a workspace');
      return;
    }
    try {
      const response = await conversationApi.create(
        currentWorkspace.id,
        'New Conversation'
      );
      setConversationId(response.data.id);
      setMessages([]);
      setError(null);
    } catch (err) {
      setError('Failed to create conversation');
      console.error(err);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    if (!currentWorkspace) {
      setError('Please select a workspace');
      return;
    }

    if (!conversationId) {
      await createNewConversation();
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setLoading(true);
    setError(null);

    // Add user message optimistically
    const tempUserMsg: Message = {
      id: Date.now().toString(),
      conversation_id: conversationId,
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    try {
      await conversationApi.sendMessage(
        conversationId,
        userMessage,
        currentWorkspace.id
      );

      // Replace temp message with real ones from server
      await loadConversation();
    } catch (err) {
      setError('Failed to send message');
      console.error(err);
      // Remove optimistic message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Chat</h1>
          {conversationId && (
            <p className="text-sm text-gray-500">ID: {conversationId.slice(0, 8)}...</p>
          )}
        </div>
        <button
          onClick={createNewConversation}
          className="btn btn-secondary flex items-center space-x-2"
        >
          <Plus size={18} />
          <span>New Chat</span>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!conversationId && messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="text-primary-600" size={32} />
              </div>
              <h2 className="text-2xl font-semibold mb-2">Start a Conversation</h2>
              <p className="text-gray-600 mb-6">
                Ask questions about your documents and get AI-powered answers with
                citations.
              </p>
              <button onClick={createNewConversation} className="btn btn-primary">
                Create New Conversation
              </button>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
              >
                <div
                  className={`max-w-3xl ${message.role === 'user'
                    ? 'bg-primary-600 text-white'
                    : 'bg-white border border-gray-200'
                    } rounded-lg p-4 shadow-sm`}
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-1">
                      {message.role === 'assistant' ? (
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      )}

                      {message.citations && message.citations.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <p className="text-xs font-medium text-gray-500 mb-2">
                            Sources:
                          </p>
                          {message.citations.map((citation, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-gray-600 mb-1 flex items-start space-x-2"
                            >
                              <span className="font-medium">[{idx + 1}]</span>
                              <span>{citation.source}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      <p className="text-xs opacity-70 mt-2">
                        {format(new Date(message.created_at), 'HH:mm')}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                  <Loader2 className="animate-spin text-primary-600" size={20} />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="px-6 py-2 bg-red-50 border-t border-red-200">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t border-gray-200 p-6">
        <div className="max-w-4xl mx-auto flex space-x-4">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your documents..."
            className="textarea flex-1"
            rows={3}
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="btn btn-primary self-end"
          >
            {loading ? (
              <Loader2 className="animate-spin" size={20} />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}