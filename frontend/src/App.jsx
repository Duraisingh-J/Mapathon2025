import { useState } from "react";
import Layout from "./components/Layout";
import FileInput from "./components/FileInput";
import Button from "./components/Button";
import ResultCard from "./components/ResultCard";
import { analyzeLake } from "./services/api";

function App() {
  const [sat, setSat] = useState(null);
  const [dem, setDem] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const getFileName = (file) => file ? file.name : "Choose file...";

  // Icons
  const UploadIcon = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  );

  const MapIcon = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0121 18.382V7.618a1 1 0 00-.553-.894L15 7m0 13V7" />
    </svg>
  );

  const AreaIcon = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
    </svg>
  );

  const VolumeIcon = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
  );

  const ErrorIcon = ({ className }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  )

  const submit = async () => {
    console.log("[DEBUG] Submit clicked. Sat:", sat, "Dem:", dem);

    if (!sat) {
      setError("Please upload a Satellite image.");
      console.warn("[DEBUG] Validation failed: No satellite image");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      console.log("[DEBUG] Calling analyzeLake service...");
      const data = await analyzeLake(sat, dem);
      console.log("[DEBUG] Analysis result:", data);

      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
    } catch (err) {
      console.error("[DEBUG] App caught error:", err);
      setError(err.message || "Failed to analyze data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 tracking-tight">
            Analyze Lake Capacity
          </h1>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg leading-relaxed">
            Upload satellite imagery and Digital Elevation Models (DEM) to calculate precise water spread area and volumetric capacity.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-white">Input Data</h3>
              <span className="text-xs font-medium px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">Satellite Required</span>
            </div>
            <div className="space-y-4">
              <FileInput
                label="Satellite Image"
                accept="image/*,.tif,.tiff"
                onChange={e => {
                  setSat(e.target.files[0]);
                  setError(null);
                }}
                icon={MapIcon}
              />
              {sat && <div className="text-xs text-green-400 flex items-center mt-1">
                <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                {sat.name}
              </div>}

              <FileInput
                label="DEM File (Optional for Volume)"
                accept=".tif,.tiff"
                onChange={e => {
                  setDem(e.target.files[0]);
                  setError(null);
                }}
                icon={MapIcon}
              />
              {dem && <div className="text-xs text-green-400 flex items-center mt-1">
                <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                {dem.name}
              </div>}
            </div>

            <div className="mt-8">
              <Button
                onClick={submit}
                disabled={loading}
                className="w-full"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing Analysis...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                    Run Analysis
                  </>
                )}
              </Button>
              {error && (
                <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-start space-x-3 text-red-400 text-sm animate-fade-in">
                  <ErrorIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {!result ? (
              <div className="h-full min-h-[300px] rounded-2xl border-2 border-dashed border-slate-700/50 bg-slate-800/20 flex flex-col items-center justify-center text-slate-500 p-8 text-center group transition-colors hover:border-slate-600/50 hover:bg-slate-800/30">
                <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                  <svg className="w-8 h-8 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-slate-400">No Results Yet</h4>
                <p className="text-sm mt-2">Upload your data and run the analysis to see water spread stats here.</p>
              </div>
            ) : (
              <div className="space-y-4 animate-fade-in">
                <ResultCard
                  title="Water Spread Area"
                  value={result.area_ha}
                  unit="Hectares"
                  icon={AreaIcon}
                  color="cyan"
                />
                <div className="grid grid-cols-2 gap-4">
                  <ResultCard
                    title="Volume (M³)"
                    value={result.volume_m3}
                    unit="m³"
                    icon={VolumeIcon}
                    color="blue"
                  />
                  <ResultCard
                    title="Volume (TMC)"
                    value={result.volume_tmc}
                    unit="TMC"
                    icon={VolumeIcon}
                    color="emerald"
                  />
                </div>

                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50 text-sm text-slate-400">
                  <p className="flex items-center">
                    <svg className="w-4 h-4 mr-2 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Analysis complete. Values are based on the provided satellite and DEM data.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;
