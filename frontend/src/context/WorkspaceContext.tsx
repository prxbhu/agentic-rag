import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Workspace, workspaceApi } from '../lib/api';

interface WorkspaceContextType {
    workspaces: Workspace[];
    currentWorkspace: Workspace | null;
    loading: boolean;
    error: string | null;
    setCurrentWorkspace: (workspace: Workspace) => void;
    createWorkspace: (name: string) => Promise<void>;
    refreshWorkspaces: () => Promise<void>;
}

const WorkspaceContext = createContext<WorkspaceContextType | undefined>(undefined);

export const WorkspaceProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
    const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchWorkspaces = async () => {
        try {
            setLoading(true);
            const response = await workspaceApi.list();
            setWorkspaces(response.data);

            // If we have workspaces but none selected (or current invalid), select the first one
            if (response.data.length > 0) {
                // Try to restore from localStorage
                const savedId = localStorage.getItem('currentWorkspaceId');
                const savedWorkspace = response.data.find(w => w.id === savedId);

                if (savedWorkspace) {
                    setCurrentWorkspace(savedWorkspace);
                } else if (!currentWorkspace) {
                    setCurrentWorkspace(response.data[0]);
                }
            } else {
                // No workspaces exist, create a default one
                await createWorkspace('Default Workspace');
            }

        } catch (err) {
            setError('Failed to fetch workspaces');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const createWorkspace = async (name: string) => {
        try {
            setLoading(true);
            const response = await workspaceApi.create(name);
            await fetchWorkspaces();
            // Auto-select the new workspace
            const newWs = response.data;
            if (newWs) {
                setCurrentWorkspace(newWs);
            }
        } catch (err) {
            setError('Failed to create workspace');
            console.error(err);
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const handleSetCurrentWorkspace = (workspace: Workspace) => {
        setCurrentWorkspace(workspace);
        localStorage.setItem('currentWorkspaceId', workspace.id);
    }

    useEffect(() => {
        fetchWorkspaces();
    }, []);

    return (
        <WorkspaceContext.Provider
            value={{
                workspaces,
                currentWorkspace,
                loading,
                error,
                setCurrentWorkspace: handleSetCurrentWorkspace,
                createWorkspace,
                refreshWorkspaces: fetchWorkspaces,
            }}
        >
            {children}
        </WorkspaceContext.Provider>
    );
};

export const useWorkspace = () => {
    const context = useContext(WorkspaceContext);
    if (context === undefined) {
        throw new Error('useWorkspace must be used within a WorkspaceProvider');
    }
    return context;
};
