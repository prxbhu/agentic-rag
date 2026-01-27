import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface HealthStatus {
  status: string;
  service?: string;
  timestamp?: string;
}

export interface Workspace {
  id: string;
  name: string;
  workspace_type: string;
  created_at: string;
}

export interface Resource {
  id: string;
  filename: string;
  workspace_id: string;
  created_at: string;
  embedding_status?: 'pending' | 'processing' | 'completed' | 'failed';
  status?: string;
}

export interface Conversation {
  id: string;
  workspace_id: string;
  title?: string;
  created_at: string;
  messages?: Message[];
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  citations?: Citation[];
}

export interface Citation {
  chunk_id: string;
  resource_id: string;
  content_preview: string;
  relevance_score?: number;
}

export interface UploadResponse {
  resource_id: string;
  task_id?: string;
  filename: string;
  status: string;
  message?: string;
}

export interface EmbeddingStatus {
  resource_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  error?: string;
}

// Health endpoints
export const healthApi = {
  check: () => api.get<HealthStatus>('/health'),
  database: () => api.get<HealthStatus>('/health/db'),
  ollama: () => api.get<HealthStatus>('/health/ollama'),
  redis: () => api.get<HealthStatus>('/health/redis'),
  embedding: () => api.get<HealthStatus>('/health/embedding'),
  detailed: () => api.get<HealthStatus>('/health/detailed'),
  ready: () => api.get<HealthStatus>('/health/ready'),
  live: () => api.get<HealthStatus>('/health/live'),
};

// Resource endpoints
export const resourceApi = {
  upload: (file: File, workspaceId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (workspaceId) {
      formData.append('workspace_id', workspaceId);
    }
    return api.post<UploadResponse>('/resources/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  list: (workspaceId?: string, limit: number = 20, offset: number = 0) => 
    api.get<{resources: Resource[]}>('/resources/list', {
        params: { workspace_id: workspaceId, limit, offset }
    }),
  getEmbeddingStatus: (resourceId: string) =>
    api.get<EmbeddingStatus>(`/resources/${resourceId}/embedding-status`),
  get: (resourceId: string) => api.get<Resource>(`/resources/${resourceId}`),
  delete: (resourceId: string) => api.delete(`/resources/${resourceId}`),
};

// Conversation endpoints
export const conversationApi = {
  create: (workspaceId: string, title?: string, systemPrompt?: string) =>
    api.post<Conversation>('/conversations', {
      workspace_id: workspaceId,
      title,
      system_prompt: systemPrompt,
    }),
  get: (conversationId: string) =>
    api.get<Conversation>(`/conversations/${conversationId}`),
  list: (workspaceId?: string, limit: number = 20, offset: number = 0) => 
    api.get<{conversations: Conversation[]}>('/conversations/list', {
      params: { workspace_id: workspaceId, limit, offset }
    }),
  delete: (conversationId: string) =>
    api.delete(`/conversations/${conversationId}`),
  sendMessage: (conversationId: string, content: string, workspaceId: string) =>
    api.post<Message>(`/conversations/${conversationId}/messages`, {
      content,
      workspace_id: workspaceId,
    }),
  export: (conversationId: string, format: 'json' | 'markdown' = 'json') =>
    api.get(`/conversations/${conversationId}/export`, {
      params: { format },
    }),
};

// Workspace endpoints
export const workspaceApi = {
  list: () => api.get<Workspace[]>('/workspaces'),
  create: (name: string, workspaceType: string = 'personal') =>
    api.post<Workspace>('/workspaces', { name, workspace_type: workspaceType }),
  get: (workspaceId: string) => api.get<Workspace>(`/workspaces/${workspaceId}`),
};

export default api;