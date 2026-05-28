import { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [redacting, setRedacting] = useState(false);
  const [findings, setFindings] = useState([]);
  const [approvedIndices, setApprovedIndices] = useState(new Set());
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setFindings([]);
    setApprovedIndices(new Set());
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8005/analyze-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to analyze PDF");

      const data = await response.json();
      setFindings(data.findings || []);

      // Auto-approve all by default
      const allIndices = new Set((data.findings || []).map((_, i) => i));
      setApprovedIndices(allIndices);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const toggleApproval = (index) => {
    const newSet = new Set(approvedIndices);
    if (newSet.has(index)) {
      newSet.delete(index);
    } else {
      newSet.add(index);
    }
    setApprovedIndices(newSet);
  };

  const handleRedactClick = async () => {
    if (!file) return;
    setRedacting(true);
    setError(null);

    try {
      // Filter out only the approved findings
      const approvedFindings = findings.filter((_, i) => approvedIndices.has(i));

      const formData = new FormData();
      formData.append("file", file);
      formData.append("findings", JSON.stringify(approvedFindings));

      const response = await fetch("http://127.0.0.1:8005/redact-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let msg = "Failed to redact PDF";
        try {
          const errData = await response.json();
          msg = errData.detail || msg;
        } catch (e) { }
        throw new Error(msg);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute("download", `redacted_${file.name}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

    } catch (error) {
      setError(error.message);
    } finally {
      setRedacting(false);
    }
  };

  return (
    <div className="container">
      <h1>VeilNet PII Redactor</h1>

      <div className="upload-section">
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button
          onClick={handleAnalyze}
          disabled={!file || analyzing}
        >
          {analyzing ? "Analyzing..." : "Analyze PDF"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {findings.length > 0 && (
        <div className="findings-section">
          <h2>Detected PII ({findings.length} findings)</h2>
          <p>Review the findings below. Uncheck any items you don't want redacted.</p>

          <div className="findings-list">
            {findings.map((f, i) => (
              <label key={i} className="finding-item">
                <input
                  type="checkbox"
                  checked={approvedIndices.has(i)}
                  onChange={() => toggleApproval(i)}
                />
                <span className="type-badge">{f.type}</span>
                <span className="value">"{f.value}"</span>
                <span className="source-badge">Pg {f.page} | {f.source}</span>
                {f.confidence && <span className="conf">{(f.confidence * 100).toFixed(1)}%</span>}
              </label>
            ))}
          </div>

          <button
            onClick={handleRedactClick}
            disabled={redacting || approvedIndices.size === 0}
            className="redact-button"
          >
            {redacting ? "Redacting..." : `Confirm & Redact ${approvedIndices.size} Items`}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
