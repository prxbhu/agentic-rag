import { useState, useRef } from 'react';
import { Upload, Loader2, CheckCircle, XCircle, AlertCircle, Trash2 } from 'lucide-react';
import { resourceApi, UploadResponse, EmbeddingStatus } from '@/lib/api';
import { useWorkspace } from '@/context/WorkspaceContext';

interface UploadedResource extends UploadResponse {
  embeddingStatus?: EmbeddingStatus;
}

export default function Resources() {
  const { currentWorkspace } = useWorkspace();
  const [resources, setResources] = useState<UploadedResource[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);

    try {
      if (!currentWorkspace) throw new Error('No workspace selected');

      const file = files[0];
      const response = await resourceApi.upload(file, currentWorkspace.id);

      const newResource = response.data;
      setResources((prev) => [newResource, ...prev]);

      // Start polling for embedding status
      pollEmbeddingStatus(newResource.resource_id);

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
            r.resource_id === resourceId
              ? { ...r, embeddingStatus: status }
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
      setResources((prev) => prev.filter((r) => r.resource_id !== resourceId));
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
    <div className="p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Resources</h1>
          <p className="text-gray-600">
            Upload and manage documents for your knowledge base
          </p>
        </div>

        {/* Upload Section */}
        <div className="card mb-8">
          <h2 className="text-xl font-semibold mb-4">Upload Document</h2>
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              className="hidden"
              accept=".pdf,.txt,.md,.doc,.docx"
              disabled={uploading}
            />
            <Upload className="mx-auto text-gray-400 mb-4" size={48} />
            <h3 className="text-lg font-medium mb-2">
              {uploading ? 'Uploading...' : 'Choose a file to upload'}
            </h3>
            <p className="text-sm text-gray-600 mb-4">
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
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Resources List */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Uploaded Documents</h2>
          {resources.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p>No documents uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-4">
              {resources.map((resource) => (
                <div
                  key={resource.resource_id}
                  className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4 flex-1">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(resource.embeddingStatus?.status)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-gray-900 truncate">
                        {resource.filename}
                      </h3>
                      <p className="text-sm text-gray-500">
                        ID: {resource.resource_id.slice(0, 8)}... •{' '}
                        {getStatusText(resource.embeddingStatus?.status)}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(resource.resource_id)}
                    className="btn btn-secondary ml-4"
                    title="Delete resource"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}