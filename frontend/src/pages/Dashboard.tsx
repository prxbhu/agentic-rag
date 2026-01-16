import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { MessageSquare, FileText, Activity, ArrowRight } from 'lucide-react';
import { healthApi } from '@/lib/api';

export default function Dashboard() {
  const [healthStatus, setHealthStatus] = useState<string>('checking');

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      await healthApi.check();
      setHealthStatus('healthy');
    } catch (error) {
      setHealthStatus('unhealthy');
    }
  };

  const quickActions = [
    {
      title: 'Start Chat',
      description: 'Begin a new conversation with the RAG assistant',
      icon: MessageSquare,
      to: '/chat',
      color: 'bg-blue-500',
    },
    {
      title: 'Upload Document',
      description: 'Add new documents to your knowledge base',
      icon: FileText,
      to: '/resources',
      color: 'bg-green-500',
    },
    {
      title: 'Check Health',
      description: 'Monitor system status and services',
      icon: Activity,
      to: '/health',
      color: 'bg-purple-500',
    },
  ];

  return (
    <div className="p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome to Agentic RAG System
          </h1>
          <p className="text-gray-600">
            Production-ready Retrieval-Augmented Generation with LangChain and LangGraph
          </p>
        </div>

        {/* System Status */}
        <div className="card mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold mb-1">System Status</h2>
              <p className="text-sm text-gray-600">
                Current operational status of the RAG system
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  healthStatus === 'healthy'
                    ? 'bg-green-500'
                    : healthStatus === 'unhealthy'
                    ? 'bg-red-500'
                    : 'bg-yellow-500'
                }`}
              />
              <span className="text-sm font-medium capitalize">{healthStatus}</span>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {quickActions.map((action) => (
              <Link
                key={action.to}
                to={action.to}
                className="card hover:shadow-md transition-shadow group"
              >
                <div className={`${action.color} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
                  <action.icon className="text-white" size={24} />
                </div>
                <h3 className="text-lg font-semibold mb-2 flex items-center justify-between">
                  {action.title}
                  <ArrowRight
                    size={18}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  />
                </h3>
                <p className="text-sm text-gray-600">{action.description}</p>
              </Link>
            ))}
          </div>
        </div>

        {/* Features */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2" />
              <div>
                <h3 className="font-medium mb-1">Query Expansion</h3>
                <p className="text-sm text-gray-600">
                  Intelligent query reformulation for better retrieval
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2" />
              <div>
                <h3 className="font-medium mb-1">Hybrid Search</h3>
                <p className="text-sm text-gray-600">
                  Combines semantic and keyword-based search
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2" />
              <div>
                <h3 className="font-medium mb-1">Multi-factor Ranking</h3>
                <p className="text-sm text-gray-600">
                  Advanced scoring for relevant context selection
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2" />
              <div>
                <h3 className="font-medium mb-1">Citation Verification</h3>
                <p className="text-sm text-gray-600">
                  Automatic citation checking and validation
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}