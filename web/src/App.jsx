// web/src/App.jsx
import React, { useEffect, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

export default function App() {
  const [runs, setRuns] = useState([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState(null);
  const [deploying, setDeploying] = useState(false);
  const [featuresText, setFeaturesText] = useState("17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189");
  const [predictions, setPredictions] = useState([]);
  const [predicting, setPredicting] = useState(false);
  const [status, setStatus] = useState("");
  const [deployedRunId, setDeployedRunId] = useState(null);

  useEffect(() => {
    fetchRuns();
  }, []);

  async function fetchRuns() {
    setLoadingRuns(true);
    setStatus("Fetching runs...");
    try {
      const res = await fetch(`${API_BASE}/runs`);
      if (!res.ok) throw new Error(`Failed to fetch runs: ${res.status}`);
      const data = await res.json();
      setRuns(data || []);
      if (data && data.length > 0) setSelectedRun(data[0].run_id);
      setStatus(`Loaded ${data ? data.length : 0} runs`);
    } catch (err) {
      console.error(err);
      setStatus(`Error fetching runs: ${err.message}`);
    } finally {
      setLoadingRuns(false);
    }
  }

  async function handleDeploy(runId) {
    setDeploying(true);
    setStatus(`Deploying ${runId}...`);
    try {
      const res = await fetch(`${API_BASE}/deploy?run_id=${encodeURIComponent(runId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Deploy failed: ${res.status} ${errText}`);
      }
      const json = await res.json();
      setDeployedRunId(json.deployed_run_id || runId);
      setStatus(`Deployed ${runId}`);
    } catch (err) {
      console.error(err);
      setStatus(`Deploy error: ${err.message}`);
    } finally {
      setDeploying(false);
    }
  }

  function parseFeatures(text) {
    // parse commas, spaces, newlines
    const parts = text.split(/[,\n\s]+/).map(s => s.trim()).filter(Boolean);
    const nums = parts.map(p => Number(p));
    if (nums.some(n => Number.isNaN(n))) throw new Error("Feature parsing produced NaN; ensure feature values are numeric and comma-separated.");
    return nums;
  }

  async function handlePredict() {
    setPredicting(true);
    setStatus("Predicting...");
    try {
      const features = parseFeatures(featuresText);
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`Predict failed: ${res.status} ${txt}`);
      }
      const json = await res.json();
      const rec = {
        timestamp: new Date().toISOString(),
        run_id: json.model_run_id || "-",
        prediction: json.prediction,
        label: json.label ?? null,
      };
      setPredictions(prev => [rec, ...prev].slice(0, 30));
      setStatus(`Prediction OK (run ${rec.run_id})`);
      setDeployedRunId(json.model_run_id || deployedRunId);
    } catch (err) {
      console.error(err);
      setStatus(`Predict error: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div style={{ fontFamily: 'Inter, Arial, sans-serif', padding: 20, maxWidth: 1100, margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
        <h1 style={{ margin: 0 }}>ModelOps Studio — Demo UI</h1>
        <div style={{ fontSize: 12, color: '#666' }}>{status}</div>
      </header>

      <section style={{ display: 'grid', gridTemplateColumns: '1fr 420px', gap: 18 }}>
        <div>
          <div style={{ marginBottom: 8, display: 'flex', gap: 8 }}>
            <button onClick={fetchRuns} disabled={loadingRuns}>Refresh runs</button>
            <button onClick={() => { if (selectedRun) handleDeploy(selectedRun); }} disabled={!selectedRun || deploying}>{deploying ? 'Deploying...' : 'Deploy selected'}</button>
            <span style={{ marginLeft: 'auto', fontSize: 12 }}>Deployed: <strong>{deployedRunId || 'none'}</strong></span>
          </div>

          <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Available runs</h3>
            {runs.length === 0 && <div style={{ color: '#666' }}>No runs found — train a model first (trainer)</div>}
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ textAlign: 'left', borderBottom: '1px solid #eee' }}>
                    <th style={{ padding: '6px 8px' }}>Run ID</th>
                    <th style={{ padding: '6px 8px' }}>Accuracy</th>
                    <th style={{ padding: '6px 8px' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => {
                    // compute once per-row (safe)
                    const isFinished = ((r.status || "").toString().toUpperCase() === "FINISHED");

                    return (
                      <tr key={r.run_id} style={{ borderBottom: '1px solid #fafafa' }}>
                        <td style={{ padding: '6px 8px' }}>
                          <label style={{ cursor: 'pointer' }}>
                            <input
                              type="radio"
                              name="selectedRun"
                              checked={selectedRun === r.run_id}
                              onChange={() => setSelectedRun(r.run_id)}
                            />
                            <span style={{ marginLeft: 8, fontFamily: 'monospace', fontSize: 12 }}>
                              {r.run_id.slice(0, 12)}
                            </span>
                          </label>
                        </td>

                        <td style={{ padding: '6px 8px' }}>
                          {r.metrics && r.metrics.test_accuracy ? Number(r.metrics.test_accuracy).toFixed(3) : '-'}
                        </td>

                        <td style={{ padding: '6px 8px' }}>
                          <button
                            onClick={() => handleDeploy(r.run_id)}
                            disabled={!isFinished || deploying}
                            title={!isFinished ? `Cannot deploy run with status ${r.status}` : `Deploy ${r.run_id}`}
                          >
                            Deploy
                          </button>
                          <a href={r.artifact_uri} style={{ marginLeft: 8 }} target="_blank" rel="noreferrer">Artifacts</a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div>
          <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 12, marginBottom: 12 }}>
            <h3 style={{ marginTop: 0 }}>Predict</h3>
            <div style={{ fontSize: 13, color: '#444', marginBottom: 6 }}>
              Paste a comma-separated feature vector (length must match the model input). A sample is provided.
            </div>
            <textarea value={featuresText} onChange={(e) => setFeaturesText(e.target.value)} rows={6} style={{ width: '100%', fontFamily: 'monospace', fontSize: 13 }} />
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              <button onClick={handlePredict} disabled={predicting}>{predicting ? 'Predicting...' : 'Predict'}</button>
              <button onClick={() => setFeaturesText('')}>Clear</button>
            </div>
          </div>

          <div style={{ border: '1px solid #eee', borderRadius: 6, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Recent predictions</h3>
            {predictions.length === 0 && <div style={{ color: '#666' }}>No predictions yet</div>}
            <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
              {predictions.map((p, idx) => (
                <li key={idx} style={{ padding: 8, borderBottom: '1px solid #fafafa' }}>
                  <div style={{ fontSize: 12, color: '#666' }}>{new Date(p.timestamp).toLocaleString()}</div>
                  <div><strong>run:</strong> <span style={{ fontFamily: 'monospace' }}>{p.run_id}</span></div>
                  <div><strong>prediction:</strong> {Number(p.prediction).toFixed(4)} {p.label !== null ? `(label ${p.label})` : ''}</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <footer style={{ marginTop: 20, color: '#888', fontSize: 13 }}>
        Tip: use the Refresh runs button after training a new model. Use Deploy to switch the live model, then Predict to test it.
      </footer>
    </div>
  );
}
