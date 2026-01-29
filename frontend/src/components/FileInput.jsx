export default function FileInput({ label, onChange, accept, icon: Icon }) {
    return (
        <div className="group">
            <label className="block text-sm font-medium text-slate-400 mb-2 transition-colors group-hover:text-cyan-400">
                {label}
            </label>
            <div className="relative">
                <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    onChange={onChange}
                    accept={accept}
                />
                <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex items-center space-x-3 transition-all group-hover:border-cyan-500/50 group-hover:bg-slate-800/80 group-focus-within:ring-2 group-focus-within:ring-cyan-500/50">
                    <div className="w-10 h-10 rounded-lg bg-slate-700/50 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 transition-colors">
                        {Icon ? <Icon className="w-5 h-5" /> : (
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                            </svg>
                        )}
                    </div>
                    <div className="flex-1 truncate">
                        <span className="text-sm text-slate-300 group-hover:text-white transition-colors">
                            Choose file...
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
