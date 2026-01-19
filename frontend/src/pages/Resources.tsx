import { useState, useRef, useEffect } from 'react';
import { Upload, Loader2, CheckCircle, XCircle, AlertCircle, Trash2 } from 'lucide-react';
import { resourceApi, UploadResponse, EmbeddingStatus, Resource } from '@/lib/api';
import { useWorkspace } from '@/context/WorkspaceContext';

interface UploadedResource extends Resource {
  embeddingStatus?: EmbeddingStatus;
}

export default function Resources() {
  const { currentWorkspace } = useWorkspace();
  const [resources, setResources] = useState<UploadedResource[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loadingList, setLoadingList] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch resources when workspace changes
  useEffect(() => {
    if (currentWorkspace) {
      // Reset list when workspace changes
      setResources([]);
      setPage(0);
      setHasMore(true);
      fetchResources(0, true);
    }
  }, [currentWorkspace]);

  const fetchResources = async (pageNum: number, reset = false) => {
    if (!currentWorkspace) return;
    setLoadingList(true);
    try {
      const response = await resourceApi.list(currentWorkspace.id, 20, pageNum * 20);
      const newResources = response.data.resources;
      
      if (newResources.length < 20) {
        setHasMore(false);
      } else {
        setHasMore(true);
      }

      if (reset) {
        setResources(newResources);
        setPage(1);
      } else {
        setResources(prev => [...prev, ...newResources]);
        setPage(pageNum + 1);
      }
    } catch (err) {
      console.error("Failed to load resources", err);
    } finally {
      setLoadingList(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);

    try {
      if (!currentWorkspace) throw new Error('No workspace selected');

      const file = files[0];
      const response = await resourceApi.upload(file, currentWorkspace.id);

      const newResource: UploadedResource = {
        id: response.data.resource_id,
        filename: response.data.filename,
        workspace_id: currentWorkspace.id,
        created_at: new Date().toISOString(),
        status: 'processing',
        embeddingStatus: { 
            resource_id: response.data.resource_id, 
            status: 'processing' 
        }
      };
      
      setResources((prev) => [newResource, ...prev]);

      // Start polling for embedding status
      pollEmbeddingStatus(newResource.id);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload file');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const pollEmbeddingStatus = async (resourceId: string) => {
    const maxAttempts = 60; // 5 minutes with 5 second intervals
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await resourceApi.getEmbeddingStatus(resourceId);
        const status = response.data;

        setResources((prev) =>
          prev.map((r) =>
            r.id === resourceId
              ? { ...r, status: status.status, embeddingStatus: status }
              : r
          )
        );

        if (status.status === 'completed' || status.status === 'failed') {
          return; // Stop polling
        }

        if (attempts < maxAttempts) {
          attempts++;
          setTimeout(poll, 5000); // Poll every 5 seconds
        }
      } catch (err) {
        console.error('Failed to poll embedding status:', err);
      }
    };

    poll();
  };

  const handleDelete = async (resourceId: string) => {
    if (!confirm('Are you sure you want to delete this resource?')) return;

    try {
      await resourceApi.delete(resourceId);
      setResources((prev) => prev.filter((r) => r.id !== resourceId));
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete resource');
      console.error(err);
    }
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="text-green-500" size={20} />;
      case 'failed':
        return <XCircle className="text-red-500" size={20} />;
      case 'processing':
        return <Loader2 className="text-blue-500 animate-spin" size={20} />;
      default:
        return <AlertCircle className="text-yellow-500" size={20} />;
    }
  };

  const getStatusText = (status?: string) => {
    switch (status) {
      case 'completed':
        return 'Ready';
      case 'failed':
        return 'Failed';
      case 'processing':
        return 'Processing';
      default:
        return 'Pending';
    }
  };

  return (
    <div className="p-8 h-full flex flex-col">
      <div className="max-w-6xl mx-auto w-full flex flex-col h-full">
        {/* Header */}
        <div className="mb-8 flex-shrink-0">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Resources</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Upload and manage documents for your knowledge base
          </p>
        </div>

        {/* Upload Section */}
        <div className="card mb-8 bg-white dark:bg-gray-800 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Upload Document</h2>
          <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center bg-gray-50 dark:bg-gray-800/50">
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              className="hidden"
              accept=".pdf,.txt,.md,.doc,.docx"
              disabled={uploading}
            />
            <Upload className="mx-auto text-gray-400 mb-4" size={48} />
            <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">
              {uploading ? 'Uploading...' : 'Choose a file to upload'}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Supported formats: PDF, TXT, MD, DOC, DOCX
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="btn btn-primary"
            >
              {uploading ? (
                <>
                  <Loader2 className="animate-spin mr-2" size={18} />
                  Uploading...
                </>
              ) : (
                'Select File'
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-8 flex-shrink-0">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {/* Resources List Container */}
        <div className="card bg-white dark:bg-gray-800 dark:border-gray-700 flex-1 flex flex-col min-h-0 overflow-hidden">
          <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white flex-shrink-0">Uploaded Documents</h2>
          
          {/* Scrollable List */}
          <div className="flex-1 overflow-y-auto min-h-0 pr-2">
            {resources.length === 0 && !loadingList ? (
                <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                <p>No documents uploaded yet</p>
                </div>
            ) : (
                <div className="space-y-4">
                {resources.map((resource) => (
                    <div
                    key={resource.id}
                    className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                    >
                    <div className="flex items-center space-x-4 flex-1">
                        <div className="flex items-center space-x-2">
                        {getStatusIcon(resource.status)}
                        </div>
                        <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-gray-900 dark:text-white truncate">
                            {resource.filename}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                            ID: {resource.id.slice(0, 8)}... •{' '}
                            {getStatusText(resource.status)}
                        </p>
                        </div>
                    </div>
                    <button
                        onClick={() => handleDelete(resource.id)}
                        className="btn btn-secondary ml-4 hover:bg-red-100 hover:text-red-600 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-red-900/30 dark:hover:text-red-400 border-none"
                        title="Delete resource"
                    >
                        <Trash2 size={18} />
                    </button>
                    </div>
                ))}
                
                {hasMore && (
                    <button 
                        onClick={() => fetchResources(page)}
                        disabled={loadingList}
                        className="w-full py-3 text-sm text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-300 disabled:opacity-50"
                    >
                        {loadingList ? (
                            <span className="flex items-center justify-center">
                                <Loader2 className="animate-spin mr-2" size={16} />
                                Loading...
                            </span>
                        ) : 'Load More'}
                    </button>
                )}
                </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}