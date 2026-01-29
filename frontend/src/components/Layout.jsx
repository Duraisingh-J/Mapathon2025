import Header from './Header';

export default function Layout({ children }) {
    return (
        <div className="min-h-screen bg-slate-900 text-slate-200 selection:bg-cyan-500/30">
            <Header />
            <main className="container mx-auto px-4 pt-24 pb-12">
                {children}
            </main>
            <footer className="py-6 text-center text-slate-500 text-sm">
                © {new Date().getFullYear()} Lake Analysis Tool. All rights reserved.
            </footer>
        </div>
    );
}
