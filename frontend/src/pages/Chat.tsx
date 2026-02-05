import { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { Send, Loader2, FileText, Plus, Settings} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { conversationApi, Message, Citation } from '@/lib/api';
import { format } from 'date-fns';
import { useWorkspace } from '@/context/WorkspaceContext';

export default function Chat() {
  const { currentWorkspace } = useWorkspace();
  const { conversationId: urlConversationId } = useParams();
  const [selectedProvider, setSelectedProvider] = useState('gemini');
  const [conversationId, setConversationId] = useState<string | null>(
    urlConversationId || null
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>(''); // For streaming status
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync state with URL param change
  useEffect(() => {
    setConversationId(urlConversationId || null);
  }, [urlConversationId]);

  useEffect(() => {
    if (conversationId) {
      loadConversation();
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, status]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadConversation = async () => {
    if (!conversationId) return;
    // Don't reload if we are currently loading/streaming to prevent overwriting
    if (loading) return; 
    
    try {
      const response = await conversationApi.get(conversationId);
      setMessages(response.data.messages || []);
    } catch (err) {
      console.error('Failed to load conversation', err);
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
        'New Conversation',
        undefined,
        selectedProvider
      );
      setConversationId(response.data.id);
      setMessages([]);
      setError(null);
      // We rely on parent navigation or just updating local state for now
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
      try {
        setLoading(true);
        const response = await conversationApi.create(
            currentWorkspace.id,
            'New Conversation',
            undefined,
            selectedProvider
        );
        const newId = response.data.id;
        setConversationId(newId);
        await performSendStream(newId, input.trim());
      } catch(err) {
        setError('Failed to start conversation');
        setLoading(false);
      }
      return;
    }

    await performSendStream(conversationId, input.trim());
  };

  const performSendStream = async (convId: string, text: string) => {
    const userMessage = text;
    setInput('');
    setLoading(true);
    setStatus('Initializing...');
    setError(null);

    // Add user message optimistically
    const tempUserMsg: Message = {
      id: Date.now().toString(),
      conversation_id: convId,
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    // Prepare placeholder for assistant message
    const tempAssistantId = (Date.now() + 1).toString();
    const tempAssistantMsg: Message = {
      id: tempAssistantId,
      conversation_id: convId,
      role: 'assistant',
      content: '', // Start empty
      created_at: new Date().toISOString(),
      citations: []
    };
    setMessages(prev => [...prev, tempAssistantMsg]);

    try {
      // Use fetch for streaming support
      const response = await fetch(`/api/conversations/${convId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: userMessage,
          workspace_id: currentWorkspace!.id
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No reader available");

      const decoder = new TextDecoder();
      let assistantContent = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // Split by newline to handle multiple JSON objects
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            
            if (data.type === 'status') {
              setStatus(data.data);
            } else if (data.type === 'content') {
              assistantContent += data.data;
              // Update message content
              setMessages(prev => 
                prev.map(m => m.id === tempAssistantId 
                  ? { ...m, content: assistantContent } 
                  : m
                )
              );
            } else if (data.type === 'end') {
              // Final update with citations
              setMessages(prev => 
                prev.map(m => m.id === tempAssistantId 
                  ? { ...m, content: assistantContent, citations: data.citations } 
                  : m
                )
              );
              setStatus('');
            } else if (data.type === 'error') {
              setError(data.data);
            }
          } catch (e) {
            console.error("Error parsing JSON chunk", e);
          }
        }
      }

    } catch (err: any) {
      setError('Failed to send message');
      console.error(err);
      // Remove optimistic assistant message on error
      setMessages((prev) => prev.filter((m) => m.id !== tempAssistantId));
    } finally {
      setLoading(false);
      setStatus('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-gray-900 dark:text-white">Chat</h1>
          {!conversationId ? (
             <div className="flex items-center mt-1 space-x-2">
                <span className="text-xs text-gray-500">Provider:</span>
                <select 
                    value={selectedProvider} 
                    onChange={(e) => setSelectedProvider(e.target.value)}
                    className="text-xs bg-gray-100 dark:bg-gray-700 border-none rounded px-2 py-1"
                >
                    <option value="gemini">Google Gemini</option>
                    <option value="vllm">vLLM(private)</option>
                    <option value="ollama">Ollama(private)</option>
                </select>
             </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">ID: {conversationId.slice(0, 8)}...</p>
          )}
        </div>
        <button
          onClick={createNewConversation}
          className="btn btn-secondary flex items-center space-x-2 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
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
              <div className="w-16 h-16 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="text-primary-600 dark:text-primary-400" size={32} />
              </div>
              <h2 className="text-2xl font-semibold mb-2 text-gray-900 dark:text-white">Start a Conversation</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
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
                    : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100'
                    } rounded-lg p-4 shadow-sm`}
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-1">
                      {message.role === 'assistant' ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      )}

                      {message.citations && message.citations.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
                            Sources:
                          </p>
                          {message.citations.map((citation, idx) => (
                            <div
                              key={idx}
                              className="text-xs text-gray-600 dark:text-gray-300 mb-1 flex items-start space-x-2"
                            >
                              <span className="font-medium">[{idx + 1}]</span>
                              <span>{citation.content_preview}</span>
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
            
            {status && (
                <div className="flex justify-start animate-pulse">
                    <div className="bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 text-xs px-3 py-1 rounded-full">
                        {status}
                    </div>
                </div>
            )}
            
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="px-6 py-2 bg-red-50 dark:bg-red-900/20 border-t border-red-200 dark:border-red-800">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Input */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-6 flex-shrink-0">
        <div className="max-w-4xl mx-auto flex space-x-4">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your documents..."
            className="textarea flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600 dark:focus:ring-primary-500"
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