import { useState, useMemo } from 'react';
import './App.css';

// SVG Icons for the UI
const Icons = {
  Document: () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
  ),
  Shield: () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
  ),
  Link: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
  ),
  Section: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 9h16"></path><path d="M4 15h16"></path><path d="M10 3L8 21"></path><path d="M16 3l-2 18"></path></svg>
  )
};

function App() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [redacting, setRedacting] = useState(false);
  
  // API Data States
  const [findings, setFindings] = useState([]);
  const [docInfo, setDocInfo] = useState(null);
  const [riskSummary, setRiskSummary] = useState(null);
  const [entityLinks, setEntityLinks] = useState([]);
  
  const [approvedIndices, setApprovedIndices] = useState(new Set());
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    // Reset states on new file
    setFindings([]);
    setDocInfo(null);
    setRiskSummary(null);
    setEntityLinks([]);
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

      const response = await fetch("http://127.0.0.1:8000/analyze-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to analyze PDF");

      const data = await response.json();
      
      setFindings(data.findings || []);
      setDocInfo(data.document_info || null);
      setRiskSummary(data.risk_summary || null);
      setEntityLinks(data.entity_links || []);

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
      // Send original index positions mapping to back-end coords
      const approvedFindings = findings.filter((_, i) => approvedIndices.has(i));

      const formData = new FormData();
      formData.append("file", file);
      formData.append("findings", JSON.stringify(approvedFindings));

      const response = await fetch("http://127.0.0.1:8000/redact-pdf", {
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
      link.setAttribute("download", `veiled_${file.name}`);
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

  // Helper to get styling for risks
  const getRiskColorClass = (risk) => {
    const rl = (risk || 'medium').toLowerCase();
    if (rl === 'critical') return 'risk-critical bg-critical';
    if (rl === 'high') return 'risk-high bg-high';
    if (rl === 'medium') return 'risk-medium bg-medium';
    if (rl === 'low') return 'risk-low bg-low';
    return 'risk-medium bg-medium';
  };

  // Sort findings by Risk Level (Critical > High > Medium > Low) then Page
  const sortedFindings = useMemo(() => {
    const riskOrder = { 'critical': 0, 'high': 1, 'medium': 2, 'low': 3 };
    return [...findings].map((f, i) => ({ ...f, originalIndex: i })).sort((a, b) => {
      const aRisk = riskOrder[(a.risk_level || 'medium').toLowerCase()] ?? 2;
      const bRisk = riskOrder[(b.risk_level || 'medium').toLowerCase()] ?? 2;
      if (aRisk !== bRisk) return aRisk - bRisk;
      return (a.page || 1) - (b.page || 1);
    });
  }, [findings]);

  return (
    <div className="container">
      <header>
        <h1>VeilNet Intelligence</h1>
        <div className="subtitle">AI-Powered PII Redaction Engine</div>
      </header>

      <section className="glass-panel upload-section">
        <div className="file-input-wrapper">
          <input type="file" accept=".pdf" onChange={handleFileChange} />
        </div>
        <button
          onClick={handleAnalyze}
          disabled={!file || analyzing}
        >
          {analyzing ? "Analyzing Document..." : "Deep Scan PDF"}
        </button>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {/* Analytics Dashboard - Appears after scan */}
      {riskSummary && docInfo && (
        <>
          <section className="dashboard-grid">
            <div className="metric-card">
              <h3>Overall Risk</h3>
              <div className={`metric-value risk-${(riskSummary.overall_risk || 'medium').toLowerCase()}`}>
                {riskSummary.overall_risk}
              </div>
            </div>
            <div className="metric-card">
              <h3>Critical PII</h3>
              <div className="metric-value risk-critical">{riskSummary.critical || 0}</div>
            </div>
            <div className="metric-card">
              <h3>High Risk</h3>
              <div className="metric-value risk-high">{riskSummary.high || 0}</div>
            </div>
            <div className="metric-card">
              <h3>Total Findings</h3>
              <div className="metric-value">{riskSummary.total || 0}</div>
            </div>
          </section>

          <section className="doc-profile">
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <Icons.Document />
              <div>
                <div className="doc-info-type">Detected as: {(docInfo.doc_type || 'Unknown').toUpperCase()}</div>
                <div className="doc-info-sub">
                  <span>Confidence: {(docInfo.doc_type_confidence * 100).toFixed(0)}%</span>
                  <span>•</span>
                  <span>{docInfo.sections_detected || 0} Semantic Sections Found</span>
                </div>
              </div>
            </div>
            <Icons.Shield />
          </section>
        </>
      )}

      {sortedFindings.length > 0 && (
        <section className="glass-panel">
          <div className="findings-header">
            <h2>Detailed Findings ({findings.length})</h2>
            <div style={{ color: 'var(--text-secondary)' }}>Uncheck to ignore</div>
          </div>

          <div className="findings-list">
            {sortedFindings.map((f) => (
              <label 
                key={f.originalIndex} 
                className={`finding-item ${approvedIndices.has(f.originalIndex) ? 'selected' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={approvedIndices.has(f.originalIndex)}
                  onChange={() => toggleApproval(f.originalIndex)}
                />
                
                <div className={`risk-badge ${getRiskColorClass(f.risk_level)}`}>
                  {f.risk_level || 'MEDIUM'}
                </div>

                <div className="value-container">
                  <span className="value">{f.value}</span>
                  <div className="context-sub">
                    <span className="type-badge" style={{ padding:'0.1rem 0.4rem', fontSize:'0.7rem', minWidth:0}}>{f.type}</span>
                    {f.section && (
                      <span title="Document Section"><Icons.Section/><span style={{marginLeft:'0.2rem'}}>{f.section}</span></span>
                    )}
                    {f.linked_entities && f.linked_entities.length > 0 && (
                      <span title={`Cross-linked with: ${f.linked_entities.join(', ')}`} style={{ color: 'var(--accent-primary)' }}>
                        <Icons.Link/><span style={{marginLeft:'0.2rem'}}>Inferred</span>
                      </span>
                    )}
                  </div>
                </div>

                <div className="meta-info">
                  <span className="source-badge">{f.source}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Pg {f.page}</span>
                </div>

                <div className="conf" style={{ minWidth: '40px', textAlign: 'right' }}>
                  {f.confidence ? `${(f.confidence * 100).toFixed(0)}%` : '--'}
                </div>
              </label>
            ))}
          </div>

          <div className="action-area">
            <button
              onClick={handleRedactClick}
              disabled={redacting || approvedIndices.size === 0}
              className="redact-button"
            >
              <Icons.Shield />
              {redacting ? "Vaulting PDF..." : `Veil ${approvedIndices.size} Entities`}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
