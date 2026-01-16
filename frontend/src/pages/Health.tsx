import { useState, useEffect } from 'react';
import { Activity, Database, Server, Box, RefreshCw, CheckCircle, XCircle } from 'lucide-react';
import { healthApi, HealthStatus } from '@/lib/api';

interface ServiceHealth {
  name: string;
  icon: React.ElementType;
  status: 'healthy' | 'unhealthy' | 'checking';
  endpoint: () => Promise<any>;
  details?: string;
}

export default function Health() {
  const [services, setServices] = useState<ServiceHealth[]>([
    {
      name: 'API Server',
      icon: Server,
      status: 'checking',
      endpoint: healthApi.check,
    },
    {
      name: 'Database',
      icon: Database,
      status: 'checking',
      endpoint: healthApi.database,
    },
    {
      name: 'Ollama',
      icon: Activity,
      status: 'checking',
      endpoint: healthApi.ollama,
    },
    {
      name: 'Redis',
      icon: Box,
      status: 'checking',
      endpoint: healthApi.redis,
    },
    {
      name: 'Embedding Service',
      icon: Activity,
      status: 'checking',
      endpoint: healthApi.embedding,
    },
  ]);
  
  const [checking, setChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  useEffect(() => {
    checkAllServices();
  }, []);

  const checkAllServices = async () => {
    setChecking(true);
    
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        try {
          const response = await service.endpoint();
          return {
            ...service,
            status: 'healthy' as const,
            details: response.data.status || 'OK',
          };
        } catch (error: any) {
          return {
            ...service,
            status: 'unhealthy' as const,
            details: error.response?.data?.detail || error.message || 'Connection failed',
          };
        }
      })
    );
    
    setServices(updatedServices);
    setLastCheck(new Date());
    setChecking(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'unhealthy':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    }
  };

  const overallHealth = services.every((s) => s.status === 'healthy')
    ? 'healthy'
    : services.some((s) => s.status === 'unhealthy')
    ? 'unhealthy'
    : 'checking';

  return (
    <div className="p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">System Health</h1>
            <p className="text-gray-600">
              Monitor the status of all system services and dependencies
            </p>
          </div>
          <button
            onClick={checkAllServices}
            disabled={checking}
            className="btn btn-primary flex items-center space-x-2"
          >
            <RefreshCw className={checking ? 'animate-spin' : ''} size={18} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Overall Status */}
        <div className={`card mb-8 ${getStatusColor(overallHealth)} border-2`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {overallHealth === 'healthy' ? (
                <CheckCircle size={32} className="text-green-600" />
              ) : overallHealth === 'unhealthy' ? (
                <XCircle size={32} className="text-red-600" />
              ) : (
                <RefreshCw size={32} className="text-yellow-600 animate-spin" />
              )}
              <div>
                <h2 className="text-2xl font-bold capitalize">{overallHealth}</h2>
                <p className="text-sm opacity-80">
                  {lastCheck
                    ? `Last checked: ${lastCheck.toLocaleTimeString()}`
                    : 'Checking...'}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium">
                {services.filter((s) => s.status === 'healthy').length} / {services.length}
              </p>
              <p className="text-xs opacity-80">Services Online</p>
            </div>
          </div>
        </div>

        {/* Service Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {services.map((service) => (
            <div
              key={service.name}
              className={`card border-2 ${getStatusColor(service.status)}`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-white rounded-lg">
                    <service.icon size={24} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg">{service.name}</h3>
                    <p className="text-sm capitalize opacity-80">{service.status}</p>
                  </div>
                </div>
                {service.status === 'healthy' ? (
                  <CheckCircle size={24} className="text-green-600" />
                ) : service.status === 'unhealthy' ? (
                  <XCircle size={24} className="text-red-600" />
                ) : (
                  <RefreshCw size={24} className="text-yellow-600 animate-spin" />
                )}
              </div>
              {service.details && (
                <div className="mt-2 pt-2 border-t border-current border-opacity-20">
                  <p className="text-sm opacity-80">{service.details}</p>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Health Check Endpoints Info */}
        <div className="card mt-8">
          <h2 className="text-xl font-semibold mb-4">Available Health Endpoints</h2>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between py-2 border-b border-gray-200">
              <span className="font-mono text-primary-600">/api/health</span>
              <span className="text-gray-600">Basic health check</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-200">
              <span className="font-mono text-primary-600">/api/health/detailed</span>
              <span className="text-gray-600">Detailed health information</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-200">
              <span className="font-mono text-primary-600">/api/health/ready</span>
              <span className="text-gray-600">Kubernetes readiness probe</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="font-mono text-primary-600">/api/health/live</span>
              <span className="text-gray-600">Kubernetes liveness probe</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}