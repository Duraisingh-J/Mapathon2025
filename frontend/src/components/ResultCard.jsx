export default function ResultCard({ title, value, unit, icon: Icon, color = "blue" }) {
    const colorStyles = {
        blue: "from-blue-500/10 to-blue-500/5 text-blue-400 border-blue-500/20",
        cyan: "from-cyan-500/10 to-cyan-500/5 text-cyan-400 border-cyan-500/20",
        emerald: "from-emerald-500/10 to-emerald-500/5 text-emerald-400 border-emerald-500/20",
    }

    return (
        <div className={`relative overflow-hidden rounded-2xl border p-6 bg-gradient-to-br ${colorStyles[color]} backdrop-blur-sm`}>
            <div className="relative z-10 flex items-start justify-between">
                <div>
                    <p className="text-sm font-medium text-slate-400 mb-1">{title}</p>
                    <h3 className="text-2xl font-bold text-white tracking-tight">
                        {value} <span className="text-sm font-normal text-slate-500 ml-1">{unit}</span>
                    </h3>
                </div>
                <div className={`p-3 rounded-xl bg-slate-900/40 border border-${color}-500/20`}>
                    {Icon && <Icon className="w-6 h-6" />}
                </div>
            </div>
        </div>
    );
}
