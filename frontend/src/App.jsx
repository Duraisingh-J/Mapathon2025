import { useState } from "react";

function App() {
  const [sat, setSat] = useState(null);
  const [dem, setDem] = useState(null);
  const [result, setResult] = useState(null);

  const submit = async () => {
    const form = new FormData();
    form.append("satellite", sat);
    form.append("dem", dem);

    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: form
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div style={{ padding: 40 }}>
      <h1>Lake Analysis Tool</h1>

      <input type="file" onChange={e => setSat(e.target.files[0])} />
      <br /><br />
      <input type="file" onChange={e => setDem(e.target.files[0])} />
      <br /><br />

      <button onClick={submit}>Analyze</button>

      {result && (
        <div>
          <h3>Results</h3>
          <p>Area: {result.area_ha} ha</p>
          <p>Volume: {result.volume_m3} m³</p>
          <p>Volume: {result.volume_tmc} TMC</p>
        </div>
      )}
    </div>
  );
}

export default App;
