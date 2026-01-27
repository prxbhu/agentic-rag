import { AlertCircle } from 'lucide-react';

interface ToastProps {
  message: string;
}

export default function Toast({ message }: ToastProps) {
  return (
    <div className="fixed top-6 right-6 z-50 animate-slide-in">
      <div className="flex items-center gap-2 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-lg">
        <AlertCircle size={18} className="text-yellow-400" />
        <span className="text-sm">{message}</span>
      </div>
    </div>
  );
}
