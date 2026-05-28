import { useState, useMemo, useEffect } from 'react';

// =============================================================================
// MOCK DATA FOR DEMO ENVIRONMENT (3 Pages of Detailed Resumes)
// =============================================================================

const DEMO_FILE_NAME = "AryanResume.pdf";

const MOCK_DOC_INFO = {
  doc_type: "resume",
  doc_type_confidence: 0.892,
  sections_detected: 5,
  sections: [
    { name: "SKILLS", sensitivity: "low" },
    { name: "PROFESSIONAL EXPERIENCE", sensitivity: "medium" },
    { name: "PROJECTS", sensitivity: "low" },
    { name: "EDUCATION", sensitivity: "medium" },
    { name: "Certifications", sensitivity: "low" }
  ]
};

const MOCK_RISK_SUMMARY = {
  overall_risk: "HIGH",
  total: 13,
  critical: 0,
  high: 5,
  medium: 8,
  low: 0,
  by_type: { "NAME": 3, "EMAIL": 3, "PHONE": 1, "LOCATION": 3, "LINKEDIN": 1, "ORG": 2 }
};

const INITIAL_MOCK_DETECTIONS = [
  // ----------- Page 1 Detections -----------
  {
    id: "det-p1-1",
    page: 1,
    type: "NAME",
    value: "Aryan Chib",
    risk_level: "high",
    source: "NER",
    confidence: 1.00,
    coords: [15, 6, 32, 11], // [Left %, Top %, Width %, Height %]
    isApproved: true
  },
  {
    id: "det-p1-2",
    page: 1,
    type: "EMAIL",
    value: "chib.aryan444@gmail.com",
    risk_level: "high",
    source: "REGEX",
    confidence: 1.00,
    coords: [15, 12.5, 33, 3],
    isApproved: true
  },
  {
    id: "det-p1-3",
    page: 1,
    type: "PHONE",
    value: "+918493069311",
    risk_level: "high",
    source: "REGEX",
    confidence: 1.00,
    coords: [50, 12.5, 20, 3],
    isApproved: true
  },
  {
    id: "det-p1-4",
    page: 1,
    type: "LINKEDIN",
    value: "linkedin.com/in/aryanchib",
    risk_level: "medium",
    source: "REGEX",
    confidence: 0.95,
    coords: [15, 16.5, 34, 3],
    isApproved: true
  },
  {
    id: "det-p1-5",
    page: 1,
    type: "LOCATION",
    value: "Jammu",
    risk_level: "medium",
    source: "NER",
    confidence: 0.99,
    coords: [74, 16.5, 10, 3],
    isApproved: true
  },
  {
    id: "det-p1-6",
    page: 1,
    type: "ORG",
    value: "Capgemini",
    risk_level: "medium",
    source: "NER",
    confidence: 1.00,
    section: "PROFESSIONAL EXPERIENCE",
    coords: [15, 29.5, 17, 3.5],
    isApproved: true
  },
  {
    id: "det-p1-7",
    page: 1,
    type: "LOCATION",
    value: "Jammu",
    risk_level: "medium",
    source: "NER",
    confidence: 0.99,
    coords: [68, 29.5, 12, 3.5],
    isApproved: true
  },
  {
    id: "det-p1-8",
    page: 1,
    type: "ORG",
    value: "Pune University",
    risk_level: "medium",
    source: "NER",
    confidence: 0.98,
    section: "EDUCATION",
    coords: [15, 59.5, 25, 3.5],
    isApproved: true
  },

  // ----------- Page 2 Detections -----------
  {
    id: "det-p2-1",
    page: 2,
    type: "NAME",
    value: "Varnit Raina",
    risk_level: "high",
    source: "NER",
    confidence: 1.00,
    coords: [15, 6, 32, 11],
    isApproved: true
  },
  {
    id: "det-p2-2",
    page: 2,
    type: "EMAIL",
    value: "varnit.raina@gmail.com",
    risk_level: "high",
    source: "REGEX",
    confidence: 1.00,
    coords: [15, 12.5, 33, 3],
    isApproved: true
  },
  {
    id: "det-p2-3",
    page: 2,
    type: "ORG",
    value: "Microsoft Corporation",
    risk_level: "medium",
    source: "NER",
    confidence: 0.97,
    coords: [15, 29.5, 34, 3.5],
    isApproved: true
  },
  {
    id: "det-p2-4",
    page: 2,
    type: "LOCATION",
    value: "Bangalore",
    risk_level: "medium",
    source: "NER",
    confidence: 0.99,
    coords: [68, 29.5, 14, 3.5],
    isApproved: true
  },

  // ----------- Page 3 Detections -----------
  {
    id: "det-p3-1",
    page: 3,
    type: "NAME",
    value: "Priya Sharma",
    risk_level: "high",
    source: "NER",
    confidence: 1.00,
    coords: [15, 6, 32, 11],
    isApproved: true
  },
  {
    id: "det-p3-2",
    page: 3,
    type: "EMAIL",
    value: "priya.sharma@yahoo.com",
    risk_level: "high",
    source: "REGEX",
    confidence: 1.00,
    coords: [15, 12.5, 33, 3],
    isApproved: true
  },
  {
    id: "det-p3-3",
    page: 3,
    type: "ORG",
    value: "Amazon Web Services",
    risk_level: "medium",
    source: "NER",
    confidence: 0.99,
    coords: [15, 29.5, 34, 3.5],
    isApproved: true
  }
];

const Icons = {
  Shield: ({ className = "w-5 h-5" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
  Upload: ({ className = "w-10 h-10" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  ),
  Cross: ({ className = "w-4 h-4" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  Sparkles: ({ className = "w-5 h-5" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  ),
  File: ({ className = "w-5 h-5" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  ArrowLeft: ({ className = "w-5 h-5" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
    </svg>
  ),
  ArrowRight: ({ className = "w-5 h-5" }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  )
};

export default function App() {
  // =============================================================================
  // APP STATE MANAGEMENT
  // =============================================================================

  // LANDING PAGE DEFAULT: Land on the Upload screen ('idle' appState, isDemoMode = false)
  const [appState, setAppState] = useState('idle');
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Multi-page page navigation state
  const [activePage, setActivePage] = useState(1);

  // Custom manual tag cloud keywords
  const [customKeywords, setCustomKeywords] = useState([]);
  const [keywordInput, setKeywordInput] = useState("");

  // Interactive click-to-redact word selection mode toggle
  const [isManualSelectMode, setIsManualSelectMode] = useState(true);

  // Core filter switches and checklist arrays
  const [enableRegex, setEnableRegex] = useState(true);
  const [enableNer, setEnableNer] = useState(true);
  const [selectedFilters, setSelectedFilters] = useState([
    "NAME", "EMAIL", "PHONE", "SSN", "CREDITCARD", "LOCATION", "ORG", "LINKEDIN"
  ]);

  const [docInfo, setDocInfo] = useState(MOCK_DOC_INFO);
  const [riskSummary, setRiskSummary] = useState(MOCK_RISK_SUMMARY);
  const [detections, setDetections] = useState(INITIAL_MOCK_DETECTIONS);
  const [activeHoverId, setActiveHoverId] = useState(null);

  // Object URLs for PDF and redacted files
  const [realFileUrl, setRealFileUrl] = useState(null);
  const [error, setError] = useState(null);
  const [redactedFileUrl, setRedactedFileUrl] = useState(null);
  const [pageTexts, setPageTexts] = useState([]);
  const [viewMode, setViewMode] = useState('pdf'); // 'pdf' | 'text'

  // Trigger Demo Sandbox environment when explicitly selected
  useEffect(() => {
    if (isDemoMode) {
      setAppState('reviewing');
      setFile({ name: DEMO_FILE_NAME, size: 62906 });
      setDetections(INITIAL_MOCK_DETECTIONS);
      setDocInfo(MOCK_DOC_INFO);
      setRiskSummary(MOCK_RISK_SUMMARY);
      setActivePage(1);
    }
  }, [isDemoMode]);

  // =============================================================================
  // INTERACTIVE WORKFLOW ACTIONS
  // =============================================================================

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    validateAndProcessFile(e.dataTransfer.files[0]);
  };

  const handleFileChange = (e) => {
    validateAndProcessFile(e.target.files[0]);
  };

  const validateAndProcessFile = (selectedFile) => {
    if (!selectedFile) return;

    if (selectedFile.type !== "application/pdf" && !selectedFile.name.endsWith(".pdf")) {
      setError("Please select a high-quality PDF document.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit.");
      return;
    }

    if (realFileUrl) {
      window.URL.revokeObjectURL(realFileUrl);
    }

    setError(null);
    setIsDemoMode(false);
    setFile(selectedFile);
    setAppState('uploading');
    setUploadProgress(0);

    const fileUrl = window.URL.createObjectURL(selectedFile);
    setRealFileUrl(fileUrl);

    // Simulated progress loading
    let progress = 0;
    const interval = setInterval(() => {
      progress += 25;
      if (progress >= 100) {
        clearInterval(interval);
        setUploadProgress(100);
        setTimeout(() => {
          setAppState('detecting');
          analyzeRealPdf(selectedFile);
        }, 400);
      } else {
        setUploadProgress(progress);
      }
    }, 70);
  };

  // Live E2E analysis coordinator
  const analyzeRealPdf = async (pdfFile) => {
    try {
      const formData = new FormData();
      formData.append("file", pdfFile);

      const response = await fetch("http://127.0.0.1:8000/analyze-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("FastAPI pipeline analysis failed.");

      const data = await response.json();

      const mapped = (data.findings || []).map((f, idx) => {
        let mockCoords = [15, 20, 30, 4];
        if (f.coords && f.coords.length === 4) {
          const xPercent = (f.coords[0] / 612) * 100;
          const yPercent = (f.coords[1] / 792) * 100;
          const wPercent = ((f.coords[2] - f.coords[0]) / 612) * 100;
          const hPercent = ((f.coords[3] - f.coords[1]) / 792) * 100;
          mockCoords = [
            Math.max(2, Math.min(xPercent, 95)),
            Math.max(2, Math.min(yPercent, 95)),
            Math.max(5, Math.min(wPercent, 90)),
            Math.max(1.5, Math.min(hPercent, 10))
          ];
        }

        return {
          id: `det-${idx}`,
          page: f.page || 1,
          type: f.type,
          value: f.value,
          risk_level: f.risk_level || "medium",
          source: f.source || "NER",
          confidence: f.confidence || 0.90,
          section: f.section,
          coords: mockCoords,
          rawCoords: f.coords,
          isApproved: true
        };
      });

      setDetections(mapped);
      setPageTexts(data.page_texts || []);
      setDocInfo(data.document_info || MOCK_DOC_INFO);
      setRiskSummary(data.risk_summary || MOCK_RISK_SUMMARY);
      setAppState('reviewing');
      setActivePage(1);
    } catch (err) {
      setError(err.message);
      setAppState('idle');
    }
  };

  // Live E2E Redaction coordinator
  const handleConfirmRedaction = async () => {
    setAppState('redacting');
    setError(null);

    const approved = detections.filter(d => d.isApproved);

    if (isDemoMode) {
      setTimeout(() => {
        setAppState('done');
      }, 1500);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const backendFindings = approved.map(d => ({
        page: d.page,
        coords: d.rawCoords,
        type: d.type,
        value: d.value
      }));
      formData.append("findings", JSON.stringify(backendFindings));

      const response = await fetch("http://127.0.0.1:8000/redact-pdf", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("FastAPI true redaction pipeline failed.");

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      setRedactedFileUrl(downloadUrl);

      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute("download", `veiled_${file.name}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);

      setAppState('done');
    } catch (err) {
      setError(err.message);
      setAppState('reviewing');
    }
  };

  const handleFilterToggle = (type) => {
    setSelectedFilters(prev =>
      prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
    );
  };

  // Custom keywords cloud tagging
  const handleAddKeyword = (e) => {
    e.preventDefault();
    if (!keywordInput.trim()) return;

    const word = keywordInput.trim();
    if (!customKeywords.includes(word)) {
      setCustomKeywords(prev => [...prev, word]);

      const randomX = Math.floor(Math.random() * 40) + 15;
      const randomY = Math.floor(Math.random() * 40) + 25;

      const newDet = {
        id: `manual-keyword-${Date.now()}`,
        page: activePage,
        type: "MANUAL",
        value: word,
        risk_level: "high",
        source: "KEYWORD",
        confidence: 1.00,
        coords: [randomX, randomY, Math.max(12, word.length * 1.6), 3.2],
        isApproved: true
      };
      setDetections(prev => [...prev, newDet]);
    }
    setKeywordInput("");
  };

  const handleRemoveKeyword = (word) => {
    setCustomKeywords(prev => prev.filter(w => w !== word));
    setDetections(prev => prev.filter(d => !(d.type === "MANUAL" && d.value === word)));
  };

  const toggleDetectionApproval = (id) => {
    setDetections(prev =>
      prev.map(d => d.id === id ? { ...d, isApproved: !d.isApproved } : d)
    );
  };

  const handleReset = () => {
    setAppState('idle');
    setFile(null);
    setDetections([]);
    setIsDemoMode(false);
    setError(null);
    setRedactedFileUrl(null);
    setCustomKeywords([]);
    setActivePage(1);
  };

  // =============================================================================
  // MEMOIZED COMPUTATIONS
  // =============================================================================

  const totalPagesCount = useMemo(() => {
    if (isDemoMode) return 3;
    const maxPage = Math.max(1, ...detections.map(d => d.page));
    return maxPage;
  }, [detections, isDemoMode]);

  // Filter coordinate detections dynamically by active page boundaries
  const filteredDetections = useMemo(() => {
    return detections.filter(d => {
      if (d.page !== activePage) return false;
      if (d.source === "REGEX" && !enableRegex) return false;
      if (d.source === "NER" && !enableNer) return false;
      if (d.type === "MANUAL" || d.type === "CUSTOM") return true;
      return selectedFilters.includes(d.type);
    });
  }, [detections, selectedFilters, enableRegex, enableNer, activePage]);

  const approvedCount = useMemo(() => {
    return detections.filter(d => d.isApproved).length;
  }, [detections]);

  // =============================================================================
  // CLICK-TO-REDACT WORD COMPONENT
  // =============================================================================

  const InteractiveWord = ({ children, section, coords }) => {
    const textVal = children.trim().replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "");
    if (!textVal || textVal.length < 2) return <span>{children} </span>;

    const isManuallyRedacted = detections.some(
      d => d.type === "MANUAL" && d.value.toLowerCase() === textVal.toLowerCase() && d.isApproved && d.page === activePage
    );

    const handleWordClick = () => {
      if (appState === 'done') return;

      const existingIdx = detections.findIndex(
        d => d.type === "MANUAL" && d.value.toLowerCase() === textVal.toLowerCase() && d.page === activePage
      );

      if (existingIdx !== -1) {
        setDetections(prev => prev.map((d, idx) => idx === existingIdx ? { ...d, isApproved: !d.isApproved } : d));
      } else {
        const newDet = {
          id: `manual-word-${Date.now()}-${Math.random()}`,
          page: activePage,
          type: "MANUAL",
          value: textVal,
          risk_level: "high",
          source: "MANUAL",
          confidence: 1.00,
          section: section,
          coords: coords || [25, 35, Math.max(10, textVal.length * 1.6), 3.2],
          isApproved: true
        };
        setDetections(prev => [...prev, newDet]);
      }
    };

    return (
      <span
        onClick={handleWordClick}
        className={`px-0.5 rounded cursor-pointer transition-all duration-150 ${isManuallyRedacted
            ? 'bg-rose-500/35 text-rose-100 border border-rose-500/50 hover:bg-rose-500/40 ring-1 ring-rose-500/30'
            : isManualSelectMode
              ? 'hover:bg-indigo-500/10 hover:text-indigo-400 hover:border-b hover:border-dashed hover:border-indigo-400'
              : ''
          }`}
        title="Click to toggle manual redaction"
      >
        {children}
      </span>
    );
  };

  const renderInteractiveText = (text, section, startCoords = [25, 35]) => {
    return text.split(" ").map((word, idx) => {
      const wordCoords = [
        startCoords[0] + (idx * 5.2),
        startCoords[1],
        Math.max(6, word.length * 1.5),
        3.2
      ];
      return (
        <InteractiveWord key={idx} section={section} coords={wordCoords}>
          {word}
        </InteractiveWord>
      );
    });
  };

  // =============================================================================
  // COMPONENT RENDERING
  // =============================================================================

  return (
    <div className="font-sans antialiased max-w-7xl mx-auto px-4 py-8 flex flex-col gap-8 w-full">

      {/* Header Logo */}
      <header className="flex flex-col md:flex-row justify-between items-center border-b border-slate-800 pb-6 gap-4 animate-fade-in">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/10 border border-indigo-500/30 rounded-lg text-indigo-400">
            <Icons.Shield className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-4xl font-display font-extrabold tracking-tight bg-gradient-to-r from-slate-100 via-indigo-200 to-violet-300 bg-clip-text text-transparent">
              VeilNet Redactor
            </h1>
            <p className="text-slate-400 text-sm mt-0.5">
              AI-Powered PII Isolation & Interactive Document Sanitization
            </p>
          </div>
        </div>

        {/* Header Action switches */}
        <div className="flex items-center gap-4">
          {appState === 'idle' ? (
            <button
              onClick={() => setIsDemoMode(true)}
              className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-300 font-semibold border border-indigo-500/30 rounded-lg transition-all duration-300 cursor-pointer shadow-lg"
            >
              <Icons.Sparkles className="w-5 h-5 text-indigo-400 animate-pulse-slow" />
              Launch Demo Sandbox
            </button>
          ) : (
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-sm transition-all duration-200 cursor-pointer"
            >
              Reset Environment
            </button>
          )}

          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-full text-xs font-semibold">
            <span className={`w-2 h-2 rounded-full ${isDemoMode ? 'bg-indigo-400 animate-ping' : 'bg-emerald-400 animate-ping'}`} />
            <span className="text-slate-400">{isDemoMode ? 'Demo Sandbox Active' : 'Live Mode Connected'}</span>
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/30 text-rose-300 px-4 py-3 rounded-lg flex items-center justify-between text-sm animate-fade-in">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200">
            <Icons.Cross className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* =============================================================================
          1. LANDING PAGE: IDLE PDF UPLOAD SCREEN (NOW SHOWN BY DEFAULT)
          ============================================================================= */}
      {appState === 'idle' && (
        <div className="flex flex-col items-center justify-center py-20 px-6 bg-slate-950/40 border border-slate-900 rounded-2xl animate-slide-up shadow-xl shadow-slate-950/50">
          <div
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className="w-full max-w-xl p-10 border border-dashed border-slate-800 hover:border-indigo-500/50 rounded-xl bg-slate-900/30 hover:bg-slate-900/50 flex flex-col items-center text-center transition-all duration-300 cursor-pointer group"
            onClick={() => document.getElementById("file-upload").click()}
          >
            <input
              id="file-upload"
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileChange}
            />
            <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-2xl text-slate-500 group-hover:text-indigo-400 group-hover:border-indigo-500/30 transition-all duration-300 mb-4 shadow-inner">
              <Icons.Upload className="w-10 h-10 group-hover:scale-110 transition-transform duration-300" />
            </div>
            <h3 className="text-slate-200 font-bold text-lg mb-1">Upload PDF Document</h3>
            <p className="text-slate-400 text-sm max-w-xs mb-6">
              Drag & drop your PDF file here, or click to browse files from your computer. (Max 10MB)
            </p>
            <div className="text-xs px-3 py-1 bg-slate-900 text-slate-500 border border-slate-800 rounded-full font-mono">
              NATIVE EMULATION LAYER READY
            </div>
          </div>
        </div>
      )}

      {/* =============================================================================
          2. LOADING & SCANNING SKELETONS
          ============================================================================= */}
      {(appState === 'uploading' || appState === 'detecting') && (
        <div className="flex flex-col items-center justify-center py-24 bg-slate-950/40 border border-slate-900 rounded-2xl animate-fade-in shadow-xl">
          <div className="relative flex items-center justify-center mb-6">
            <div className="w-16 h-16 border-4 border-indigo-500/10 border-t-indigo-500 rounded-full animate-spin" />
            <Icons.Shield className="w-6 h-6 text-indigo-400 absolute animate-pulse" />
          </div>
          <h3 className="text-slate-200 font-bold text-xl mb-2 font-display">
            {appState === 'uploading' ? 'Vaulting Document Safely...' : 'Extracting & Scanning PII Targets...'}
          </h3>
          <p className="text-slate-400 text-sm max-w-sm text-center mb-6 leading-relaxed">
            {appState === 'uploading'
              ? 'Securing file pipeline and preparing document for server classification.'
              : 'Agent 3 & Agent 5 are matching regex strings, evaluating NER, and calculating risk profiles.'}
          </p>
          {appState === 'uploading' && (
            <div className="w-64 h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div
                className="h-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-200"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* =============================================================================
          3. MAIN SPLIT-PANE DASHBOARD PANELS
          ============================================================================= */}
      {(appState === 'reviewing' || appState === 'redacting' || appState === 'done') && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 w-full animate-slide-up">

          {/* ==================== LEFT CONTROL SIDEBAR ==================== */}
          <aside className="lg:col-span-4 flex flex-col gap-6 lg:sticky lg:top-8 h-fit">

            {/* Metadata Card */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-5 shadow-lg shadow-slate-950/40">
              <div className="flex items-center gap-3 mb-3">
                <Icons.File className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-slate-200 text-sm truncate max-w-xs" title={file?.name}>
                  {file?.name || DEMO_FILE_NAME}
                </h3>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4 border-t border-slate-900 pt-3">
                <div>
                  <span className="text-[10px] text-slate-500 block uppercase font-mono tracking-wider">Classification</span>
                  <span className="text-slate-200 text-sm font-semibold capitalize font-display">
                    {docInfo?.doc_type || 'resume'}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block uppercase font-mono tracking-wider">Doc Confidence</span>
                  <span className="text-slate-200 text-sm font-semibold font-mono">
                    {((docInfo?.doc_type_confidence || 0.89) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Dynamic Risk Meter */}
              <div className="mt-4 border-t border-slate-900 pt-3">
                <div className="flex justify-between text-xs mb-1 font-semibold">
                  <span className="text-slate-400 font-display">Threat Score Level</span>
                  <span className={`font-bold ${riskSummary?.overall_risk === 'HIGH' ? 'text-rose-400' : 'text-amber-400'
                    }`}>{riskSummary?.overall_risk || 'MEDIUM'} RISK</span>
                </div>
                <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${riskSummary?.overall_risk === 'HIGH' ? 'bg-rose-500' : 'bg-amber-500'
                      }`}
                    style={{ width: riskSummary?.overall_risk === 'HIGH' ? '85%' : '50%' }}
                  />
                </div>
              </div>
            </div>

            {/* Click-to-Redact Manual Mode toggle */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-5 shadow-lg shadow-slate-950/40">
              <h4 className="text-xs uppercase font-mono tracking-widest text-slate-400 mb-3 border-b border-slate-900 pb-2">
                Interactive Redaction Mode
              </h4>
              <div className="flex items-center justify-between cursor-pointer group">
                <div className="flex flex-col">
                  <span className="text-slate-200 text-sm font-semibold group-hover:text-indigo-300 transition-colors">
                    Click-to-Select Words
                  </span>
                  <span className="text-[10px] text-slate-500">Toggle manual highlights directly on text</span>
                </div>
                <input
                  type="checkbox"
                  checked={isManualSelectMode}
                  onChange={(e) => setIsManualSelectMode(e.target.checked)}
                  className="w-4.5 h-4.5 accent-indigo-500 cursor-pointer"
                />
              </div>
            </div>

            {/* PII Filters checklist */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-5 shadow-lg shadow-slate-950/40">
              <div className="flex justify-between items-center mb-4 border-b border-slate-900 pb-2">
                <h4 className="text-xs uppercase font-mono tracking-widest text-slate-400">
                  PII Entity Filters
                </h4>
                <button
                  onClick={() => setSelectedFilters(
                    selectedFilters.length > 0 ? [] : ["NAME", "EMAIL", "PHONE", "SSN", "CREDITCARD", "LOCATION", "ORG", "LINKEDIN"]
                  )}
                  className="text-[10px] text-indigo-400 hover:text-indigo-300 font-semibold cursor-pointer"
                >
                  {selectedFilters.length > 0 ? 'Clear All' : 'Select All'}
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Person", value: "NAME", color: "bg-indigo-500" },
                  { label: "Emails", value: "EMAIL", color: "bg-sky-500" },
                  { label: "Phones", value: "PHONE", color: "bg-amber-500" },
                  { label: "SSN", value: "SSN", color: "bg-rose-500" },
                  { label: "Cards", value: "CREDITCARD", color: "bg-emerald-500" },
                  { label: "Location", value: "LOCATION", color: "bg-teal-500" },
                  { label: "Company", value: "ORG", color: "bg-purple-500" },
                  { label: "LinkedIn", value: "LINKEDIN", color: "bg-blue-500" }
                ].map((item) => (
                  <label
                    key={item.value}
                    className="flex items-center gap-2 cursor-pointer group text-xs text-slate-300 hover:text-slate-100 transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={selectedFilters.includes(item.value)}
                      onChange={() => handleFilterToggle(item.value)}
                      className="w-3.5 h-3.5 accent-indigo-500 cursor-pointer"
                    />
                    <span className={`w-1.5 h-1.5 rounded-full ${item.color}`} />
                    <span>{item.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Custom tags keyword adder */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-5 shadow-lg shadow-slate-950/40">
              <h4 className="text-xs uppercase font-mono tracking-widest text-slate-400 mb-3 border-b border-slate-900 pb-2">
                Custom PII Keywords
              </h4>
              <form onSubmit={handleAddKeyword} className="flex gap-2">
                <input
                  type="text"
                  value={keywordInput}
                  onChange={(e) => setKeywordInput(e.target.value)}
                  placeholder="e.g. Secret, IP, Code..."
                  className="flex-grow px-3 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 transition-colors"
                />
                <button
                  type="submit"
                  className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold transition-all cursor-pointer"
                >
                  Add
                </button>
              </form>

              {customKeywords.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-4">
                  {customKeywords.map((word) => (
                    <span
                      key={word}
                      className="flex items-center gap-1.5 pl-2.5 pr-1.5 py-1 bg-slate-900 hover:bg-slate-805 border border-slate-800 text-[11px] font-semibold text-indigo-300 rounded-full"
                    >
                      {word}
                      <button
                        type="button"
                        onClick={() => handleRemoveKeyword(word)}
                        className="text-slate-500 hover:text-rose-400 cursor-pointer"
                      >
                        <Icons.Cross className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Engines config */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-5 shadow-lg shadow-slate-950/40">
              <h4 className="text-xs uppercase font-mono tracking-widest text-slate-400 mb-4 border-b border-slate-900 pb-2">
                Scanning Engines
              </h4>
              <div className="flex flex-col gap-3">
                <label className="flex items-center justify-between cursor-pointer group">
                  <div className="flex flex-col">
                    <span className="text-slate-200 text-sm font-semibold group-hover:text-indigo-300 transition-colors">
                      Regex Engine
                    </span>
                    <span className="text-[10px] text-slate-500">Fast pattern-matching</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={enableRegex}
                    onChange={(e) => setEnableRegex(e.target.checked)}
                    className="w-4 h-4 accent-indigo-500 cursor-pointer"
                  />
                </label>

                <label className="flex items-center justify-between cursor-pointer group">
                  <div className="flex flex-col">
                    <span className="text-slate-200 text-sm font-semibold group-hover:text-indigo-300 transition-colors">
                      NER Engine
                    </span>
                    <span className="text-[10px] text-slate-500">Transformer deep-learning</span>
                  </div>
                  <input
                    type="checkbox"
                    checked={enableNer}
                    onChange={(e) => setEnableNer(e.target.checked)}
                    className="w-4 h-4 accent-indigo-500 cursor-pointer"
                  />
                </label>
              </div>
            </div>

          </aside>

          {/* ==================== RIGHT VIEWING PANELS ==================== */}
          <main className="lg:col-span-8 flex flex-col gap-6">

            {/* Top Redaction Bar Dashboard */}
            <div className="bg-slate-950/60 border border-slate-900 rounded-xl p-4 flex flex-col md:flex-row justify-between items-center gap-4 shadow-lg shadow-slate-950/40">
              <div className="flex items-center gap-4">
                <div className="px-3 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg text-indigo-400 text-xs font-mono">
                  Detections (Active Pg): {filteredDetections.length}
                </div>
                <div className="px-3 py-1.5 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-xs font-mono">
                  Redacted (Total): {approvedCount}
                </div>
              </div>

              <div className="flex items-center gap-3">
                {appState === 'done' && redactedFileUrl && (
                  <a
                    href={redactedFileUrl}
                    download={`redacted_${file?.name || 'document.pdf'}`}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-all duration-200 cursor-pointer"
                  >
                    Download Redacted PDF
                  </a>
                )}
                <button
                  onClick={handleConfirmRedaction}
                  disabled={approvedCount === 0 || appState === 'redacting' || appState === 'done'}
                  className="flex items-center gap-2 px-6 py-2.5 bg-rose-600 hover:bg-rose-500 disabled:bg-slate-900 disabled:text-slate-600 disabled:border-slate-800 disabled:cursor-not-allowed text-white font-semibold rounded-lg text-sm transition-all duration-200 cursor-pointer border border-rose-500/20 shadow-lg shadow-rose-950/20"
                >
                  <Icons.Shield className="w-4 h-4" />
                  {appState === 'redacting' ? 'Vaulting PDF...' : appState === 'done' ? 'Redaction Applied' : `Confirm & Redact ${approvedCount} Items`}
                </button>
              </div>
            </div>

            {/* Document Bounding Canvas Container */}
            <div className="bg-slate-950/40 border border-slate-900 rounded-2xl p-6 md:p-8 flex flex-col gap-6 items-center shadow-inner relative overflow-hidden">

              {/* Radial glow background */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-indigo-500/5 rounded-full blur-[100px] pointer-events-none" />

              {/* Dynamic Page Navigation Controls */}
              <div className="z-10 flex items-center justify-between w-full max-w-[650px] bg-slate-950 border border-slate-900 px-4 py-2 rounded-xl text-xs font-mono shadow-md gap-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActivePage(prev => Math.max(1, prev - 1))}
                    disabled={activePage === 1}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-900 border border-slate-800 hover:bg-slate-805 disabled:bg-slate-950 disabled:border-slate-900 disabled:text-slate-700 text-slate-300 rounded-lg transition-all duration-150 cursor-pointer"
                  >
                    <Icons.ArrowLeft className="w-3.5 h-3.5" />
                  </button>

                  <span className="text-slate-300 font-bold px-1">
                    Pg {activePage} of {totalPagesCount}
                  </span>

                  <button
                    onClick={() => setActivePage(prev => Math.min(totalPagesCount, prev + 1))}
                    disabled={activePage === totalPagesCount}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-900 border border-slate-800 hover:bg-slate-805 disabled:bg-slate-950 disabled:border-slate-900 disabled:text-slate-700 text-slate-300 rounded-lg transition-all duration-150 cursor-pointer"
                  >
                    <Icons.ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* View Mode Toggle (Only visible in Connected Mode when PDF is uploaded) */}
                {!isDemoMode && (
                  <div className="flex items-center bg-slate-900 border border-slate-800 p-0.5 rounded-lg">
                    <button
                      onClick={() => setViewMode('pdf')}
                      className={`px-3 py-1.5 rounded-md transition-all duration-200 font-sans font-semibold cursor-pointer ${viewMode === 'pdf' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                        }`}
                    >
                      📄 Visual PDF
                    </button>
                    <button
                      onClick={() => setViewMode('text')}
                      className={`px-3 py-1.5 rounded-md transition-all duration-200 font-sans font-semibold cursor-pointer ${viewMode === 'text' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                        }`}
                    >
                      🔍 Interactive Selector
                    </button>
                  </div>
                )}
              </div>

              {/* =============================================================================
                  DOCUMENT VIEW CANVAS: NOW POSITIONED ABSOLUTE INTERNALLY FOR PERFECT BOUNDS
                  ============================================================================= */}
              <div className={`w-full max-w-[650px] aspect-[1/1.4] bg-white text-slate-800 rounded-lg shadow-2xl relative font-serif select-none transition-all duration-500 border border-slate-200 ${isDemoMode ? 'p-8' : 'p-0'}`}>

                {/* Visual blackout layer for redacted state */}
                {appState === 'done' && (
                  <div className="absolute inset-0 bg-slate-900/10 backdrop-blur-[0.5px] rounded-lg z-10 pointer-events-none transition-all duration-1000" />
                )}

                {isDemoMode ? (
                  /* ==========================================
                     Simulated HTML Multi-page Content blocks
                     ========================================== */
                  <div className="flex flex-col gap-6 w-full h-full opacity-90 animate-fade-in text-xs font-sans">

                    {/* Render Page 1 contents */}
                    {activePage === 1 && (
                      <div className="flex flex-col gap-5 w-full h-full animate-fade-in">
                        <div className="text-center border-b pb-4">
                          <h2 className="text-2xl font-bold font-sans tracking-tight text-slate-900">
                            {renderInteractiveText("Aryan Chib", "HEADER", [15, 6])}
                          </h2>
                          <p className="text-xs text-slate-500 font-sans tracking-wider mt-1.5 flex justify-center gap-4">
                            <span>{renderInteractiveText("chib.aryan444@gmail.com", "CONTACT", [15, 12.5])}</span>
                            <span>•</span>
                            <span>{renderInteractiveText("+918493069311", "CONTACT", [50, 12.5])}</span>
                          </p>
                          <p className="text-xs text-slate-500 font-sans tracking-wider mt-0.5 flex justify-center gap-4">
                            <span>{renderInteractiveText("linkedin.com/in/aryanchib", "CONTACT", [15, 16.5])}</span>
                            <span>•</span>
                            <span>{renderInteractiveText("Jammu", "CONTACT", [74, 16.5])}</span>
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-1">
                            Professional Summary
                          </h4>
                          <p className="text-slate-600">
                            {renderInteractiveText("Result-driven software developer with extensive background building modern AI pipeline tools, redactor wrappers, and microservices architecture.", "SUMMARY")}
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-2 border-b pb-1">
                            Professional Experience
                          </h4>
                          <div className="flex justify-between items-start mb-1 font-semibold text-slate-900">
                            <span>{renderInteractiveText("Software Engineer — Capgemini", "EXPERIENCE")}</span>
                            <span className="text-[10px] text-slate-500 font-normal">Jammu | 2024 - Present</span>
                          </div>
                          <ul className="list-disc pl-4 text-slate-600 flex flex-col gap-1">
                            <li>{renderInteractiveText("Developed a multi-agent orchestration service that performs text parsing and regex detection.", "EXPERIENCE")}</li>
                            <li>{renderInteractiveText("Configured document classifier pipeline reducing false positive rates by 23% across location filters.", "EXPERIENCE")}</li>
                          </ul>
                        </div>
                      </div>
                    )}

                    {/* Render Page 2 contents */}
                    {activePage === 2 && (
                      <div className="flex flex-col gap-5 w-full h-full animate-fade-in">
                        <div className="text-center border-b pb-4">
                          <h2 className="text-2xl font-bold font-sans tracking-tight text-slate-900">
                            {renderInteractiveText("Varnit Raina", "HEADER", [15, 6])}
                          </h2>
                          <p className="text-xs text-slate-500 font-sans tracking-wider mt-1.5 flex justify-center gap-4">
                            <span>{renderInteractiveText("varnit.raina@gmail.com", "CONTACT", [15, 12.5])}</span>
                            <span>•</span>
                            <span>+91-98765-43210</span>
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-1">
                            Core Competencies
                          </h4>
                          <p className="text-slate-600">
                            {renderInteractiveText("Intelligent enterprise design patterns, cloud computing architectures, and automatic NER entity classification frameworks.", "SUMMARY")}
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-2 border-b pb-1">
                            Professional Experience
                          </h4>
                          <div className="flex justify-between items-start mb-1 font-semibold text-slate-900">
                            <span>{renderInteractiveText("Senior Solutions Architect — Microsoft Corporation", "EXPERIENCE", [15, 29.5])}</span>
                            <span className="text-[10px] text-slate-500 font-normal">Bangalore | 2022 - 2024</span>
                          </div>
                          <ul className="list-disc pl-4 text-slate-600 flex flex-col gap-1">
                            <li>{renderInteractiveText("Designed PII identification architectures to secure internal data lakes.", "EXPERIENCE")}</li>
                            <li>{renderInteractiveText("Supervised multi-model ensemble filters to deduplicate cross-linked entities.", "EXPERIENCE")}</li>
                          </ul>
                        </div>
                      </div>
                    )}

                    {/* Render Page 3 contents */}
                    {activePage === 3 && (
                      <div className="flex flex-col gap-5 w-full h-full animate-fade-in">
                        <div className="text-center border-b pb-4">
                          <h2 className="text-2xl font-bold font-sans tracking-tight text-slate-900">
                            {renderInteractiveText("Priya Sharma", "HEADER", [15, 6])}
                          </h2>
                          <p className="text-xs text-slate-500 font-sans tracking-wider mt-1.5 flex justify-center gap-4">
                            <span>{renderInteractiveText("priya.sharma@yahoo.com", "CONTACT", [15, 12.5])}</span>
                            <span>•</span>
                            <span>+91-88888-99999</span>
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-1">
                            Profile Summary
                          </h4>
                          <p className="text-slate-600">
                            {renderInteractiveText("Deep learning engineer focusing on transformers architectures, contextual word embeddings, and text classification models.", "SUMMARY")}
                          </p>
                        </div>

                        <div>
                          <h4 className="font-bold text-slate-900 uppercase tracking-wider text-[11px] mb-2 border-b pb-1">
                            Employment History
                          </h4>
                          <div className="flex justify-between items-start mb-1 font-semibold text-slate-900">
                            <span>{renderInteractiveText("Deep Learning Engineer — Amazon Web Services", "EXPERIENCE", [15, 29.5])}</span>
                            <span className="text-[10px] text-slate-500 font-normal">Pune | 2021 - 2023</span>
                          </div>
                        </div>
                      </div>
                    )}

                  </div>
                ) : (
                  /* Real PDF View Canvas (Visual PDF vs. Interactive Selector) */
                  <div className="w-full h-full rounded-lg overflow-hidden animate-fade-in bg-slate-900">
                    {realFileUrl ? (
                      viewMode === 'pdf' ? (
                        <embed
                          src={`${realFileUrl}#page=${activePage}&toolbar=0&navpanes=0&scrollbar=0`}
                          type="application/pdf"
                          className="w-full h-full rounded-lg border-0"
                        />
                      ) : (
                        /* Interactive Text Selector View */
                        <div className="w-full h-full p-8 bg-white text-slate-800 overflow-y-auto select-text font-sans text-sm leading-relaxed flex flex-col gap-4 text-left">
                          <h4 className="text-xs uppercase font-mono tracking-widest text-slate-400 mb-2 border-b pb-2">
                            Interactive Reader — Click words to redact
                          </h4>
                          {pageTexts[activePage - 1] ? (
                            pageTexts[activePage - 1].split('\n').map((paragraph, pIdx) => {
                              if (!paragraph.trim()) return null;
                              return (
                                <p key={pIdx} className="text-slate-700">
                                  {renderInteractiveText(paragraph, "MANUAL", [15, 20 + (pIdx * 5)])}
                                </p>
                              );
                            })
                          ) : (
                            <div className="text-slate-400 text-center py-10">
                              Scanning page text... Or no text found on this page.
                            </div>
                          )}
                        </div>
                      )
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm font-sans">
                        No active PDF document loaded.
                      </div>
                    )}
                  </div>
                )}

                {/* =============================================================================
                    DYNAMIC BOUNDING BOX OVERLAYS MAPPED BY ACTIVE PAGE (NOW CORRECTLY NESTED INSIDE DOCUMENT)
                    ============================================================================= */}
                {(isDemoMode || viewMode === 'pdf') && filteredDetections.map((det) => {
                  const isRedacted = appState === 'done';

                  return (
                    <div
                      key={det.id}
                      className="absolute z-20 group/highlight animate-fade-in"
                      style={{
                        left: `${det.coords[0]}%`,
                        top: `${det.coords[1]}%`,
                        width: `${det.coords[2]}%`,
                        height: `${det.coords[3]}%`
                      }}
                      onMouseEnter={() => setActiveHoverId(det.id)}
                      onMouseLeave={() => setActiveHoverId(null)}
                    >
                      {/* Bounding Highlight Box */}
                      <div
                        className={`w-full h-full rounded transition-all duration-300 ${isRedacted && det.isApproved
                            ? 'bg-slate-950 border border-slate-950 scale-x-105 shadow-lg'
                            : det.isApproved
                              ? 'bg-rose-500/20 border-2 border-rose-500/60 ring-2 ring-rose-500/20 scale-102 hover:scale-105 cursor-pointer'
                              : 'bg-slate-300/10 border border-slate-300/30 line-through opacity-40 hover:opacity-80 cursor-pointer'
                          }`}
                        onClick={() => !isRedacted && toggleDetectionApproval(det.id)}
                      />

                      {det.isApproved && !isRedacted && (
                        <span className="absolute -top-3.5 -left-1 px-1.5 py-0.5 bg-rose-500 text-white font-mono text-[8px] font-bold uppercase rounded scale-75 origin-bottom-left shadow pointer-events-none">
                          {det.type}
                        </span>
                      )}

                      {/* Popover details */}
                      {activeHoverId === det.id && !isRedacted && (
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-slate-950 border border-slate-800 text-slate-100 rounded-xl p-3 shadow-2xl z-30 animate-fade-in flex flex-col gap-2 font-sans text-left">
                          <div className="flex justify-between items-center">
                            <span className="text-[9px] px-1.5 py-0.5 bg-indigo-500/10 border border-indigo-500/20 rounded font-semibold text-indigo-300 font-mono tracking-wider">
                              {det.type}
                            </span>
                            <span className="text-[10px] text-slate-500 font-mono">
                              {(det.confidence * 100).toFixed(0)}% conf
                            </span>
                          </div>
                          <div className="text-xs font-semibold truncate border-b border-slate-900 pb-1.5 text-slate-200">
                            "{det.value}"
                          </div>
                          <div className="flex justify-between items-center text-[10px] text-slate-400">
                            <span>Source: <strong className="font-mono text-indigo-400">{det.source}</strong></span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleDetectionApproval(det.id);
                              }}
                              className={`px-2 py-0.5 rounded font-bold transition-all cursor-pointer ${det.isApproved
                                  ? 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/20'
                                  : 'bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-400 border border-indigo-500/20'
                                }`}
                            >
                              {det.isApproved ? 'Exclude' : 'Include'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

            </div>

          </main>

        </div>
      )}

    </div>
  );
}
