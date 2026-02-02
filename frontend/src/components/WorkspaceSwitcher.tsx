import React, { useState } from 'react';
import { useWorkspace } from '../context/WorkspaceContext';
import { ChevronDown, Plus, Check, Search } from 'lucide-react';

export default function WorkspaceSwitcher() {
    const { workspaces, currentWorkspace, setCurrentWorkspace, createWorkspace, loading } = useWorkspace();
    const [isOpen, setIsOpen] = useState(false);
    const [isCreating, setIsCreating] = useState(false);
    const [newWorkspaceName, setNewWorkspaceName] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    const handleCreate = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!newWorkspaceName.trim()) return;

        await createWorkspace(newWorkspaceName);
        setNewWorkspaceName('');
        setIsCreating(false);
        setIsOpen(false);
    };

    const filteredWorkspaces = workspaces.filter(ws => 
        ws.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading || !currentWorkspace) {
        return <div className="text-sm text-gray-400">Loading...</div>;
    }

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center space-x-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-sm font-medium transition-colors w-full border border-transparent dark:text-gray-200"
            >
                <div className="w-6 h-6 rounded bg-primary-600 flex items-center justify-center text-xs text-white uppercase">
                    {currentWorkspace.name.substring(0, 2)}
                </div>
                <span className="flex-1 text-left truncate">{currentWorkspace.name}</span>
                <ChevronDown size={16} className={`text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <>
                    <div
                        className="fixed inset-0 z-10"
                        onClick={() => {
                            setIsOpen(false);
                            setIsCreating(false);
                            setSearchQuery('');
                        }}
                    />
                    <div className="absolute top-full left-0 mt-1 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-20 overflow-hidden">
                        
                        <div className="p-2 border-b border-gray-200 dark:border-gray-700">
                            <div className="flex items-center px-2 bg-gray-100 dark:bg-gray-700 rounded-md">
                                <Search size={14} className="text-gray-400 mr-2" />
                                <input 
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder="Search workspaces..."
                                    className="w-full bg-transparent border-none text-sm px-0 py-1 focus:ring-0 text-gray-900 dark:text-white focus:outline-none"
                                    autoFocus
                                />
                            </div>
                        </div>

                        <div className="p-2 space-y-1 max-h-60 overflow-y-auto">
                            <div className="px-3 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                Workspaces
                            </div>
                            {filteredWorkspaces.map((ws) => (
                                <button
                                    key={ws.id}
                                    onClick={() => {
                                        setCurrentWorkspace(ws);
                                        setIsOpen(false);
                                        setSearchQuery('');
                                    }}
                                    className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded-md transition-colors ${ws.id === currentWorkspace.id
                                            ? 'bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-400'
                                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                                        }`}
                                >
                                    <span className="truncate">{ws.name}</span>
                                    {ws.id === currentWorkspace.id && <Check size={14} />}
                                </button>
                            ))}
                            {filteredWorkspaces.length === 0 && (
                                <div className="px-3 py-2 text-xs text-gray-500 dark:text-gray-400 text-center">
                                    No workspaces found
                                </div>
                            )}
                        </div>

                        <div className="border-t border-gray-200 dark:border-gray-700 p-2">
                            {isCreating ? (
                                <form onSubmit={handleCreate} className="space-y-2">
                                    <input
                                        type="text"
                                        value={newWorkspaceName}
                                        onChange={(e) => setNewWorkspaceName(e.target.value)}
                                        placeholder="Workspace name"
                                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500 text-gray-900 dark:text-white dark:bg-gray-700"
                                        autoFocus
                                    />
                                    <div className="flex space-x-2">
                                        <button
                                            type="submit"
                                            disabled={!newWorkspaceName.trim()}
                                            className="flex-1 px-3 py-1.5 bg-primary-600 text-white text-xs font-medium rounded hover:bg-primary-700 disabled:opacity-50"
                                        >
                                            Create
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setIsCreating(false)}
                                            className="px-3 py-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs font-medium rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </form>
                            ) : (
                                <button
                                    onClick={() => setIsCreating(true)}
                                    className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-md transition-colors"
                                >
                                    <Plus size={16} />
                                    <span>New Workspace</span>
                                </button>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}