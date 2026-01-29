export default function Button({ children, onClick, disabled, variant = 'primary', className = '' }) {
    const baseStyles = "relative inline-flex items-center justify-center px-6 py-3 overflow-hidden font-medium transition-all rounded-xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed";

    const variants = {
        primary: "bg-gradient-to-r from-blue-500 to-cyan-400 text-white hover:from-blue-600 hover:to-cyan-500 shadow-lg shadow-cyan-500/25 focus:ring-cyan-400",
        secondary: "bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700 focus:ring-slate-500",
    };

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`${baseStyles} ${variants[variant]} ${className}`}
        >
            <span className="relative flex items-center gap-2">
                {children}
            </span>
        </button>
    );
}
