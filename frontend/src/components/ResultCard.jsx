export default function ResultCard({ title, value, unit, icon: Icon, color = "cyan", ...props }) {
    const colorClasses = {
        cyan: "from-cyan-500 to-blue-500 text-cyan-400",
        blue: "from-blue-500 to-indigo-500 text-blue-400",
        emerald: "from-emerald-500 to-teal-500 text-emerald-400",
    };

    const gradient = colorClasses[color] || colorClasses.cyan;

    return (
        <div className="bg-slate-800 rounded px-4 py-4 border-l-2 border-slate-700 hover:border-l-cyan-500 transition-colors group relative">
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-1">{title}</p>
                    <div className="flex items-baseline space-x-2">
                        <span className="text-2xl font-bold text-white font-mono">{value != null ? value : "-"}</span>
                        <span className="text-xs text-slate-500">{unit}</span>
                    </div>
                    {/* Optional Subtext (e.g. % change) */}
                    {props.subText && (
                        <div className={`text-[10px] mt-1 font-mono font-bold ${props.subColor || "text-slate-500"}`}>
                            {props.subText}
                        </div>
                    )}
                </div>
                <div className={`p-2 rounded bg-slate-900/50 text-${color}-500 opacity-75 group-hover:opacity-100 group-hover:text-${color}-400 transition-all`}>
                    <Icon className="w-5 h-5" />
                </div>
            </div>
        </div>
    );
}
