import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [darkMode, setDarkMode] = useState(false);

  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem("voiceshield-history");
    try {
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  // Remember login and theme
  useEffect(() => {
    const savedLogin = localStorage.getItem("voiceshield-login");
    const savedTheme = localStorage.getItem("voiceshield-theme");

    if (savedLogin) {
      setLoggedIn(true);
    }

    if (savedTheme === "dark") {
      setDarkMode(true);
    }
  }, []);

  // Sync history to localStorage
  useEffect(() => {
    localStorage.setItem("voiceshield-history", JSON.stringify(history));
  }, [history]);

  // Theme
  useEffect(() => {
    document.body.className = darkMode ? "dark" : "";

    localStorage.setItem(
      "voiceshield-theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  // Login
  const handleLogin = (e) => {
    e.preventDefault();

    if (!email || !password) {
      alert("Please enter email and password.");
      return;
    }

    localStorage.setItem("voiceshield-login", "true");
    setLoggedIn(true);
  };

  // Logout
  const handleLogout = () => {
    localStorage.removeItem("voiceshield-login");

    setLoggedIn(false);
    setFile(null);
    setResult(null);
    setEmail("");
    setPassword("");
  };

  // File selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    localStorage.removeItem("voiceshield-history");
  };

  // Analyze audio via backend API
  const handleAnalyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("audio", file);

      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const message =
          errorData?.detail || `Server error (${response.status})`;
        throw new Error(message);
      }

      const analysisResult = await response.json();

      setResult(analysisResult);

      // Determine confidence percentage safely
      const confidence = analysisResult.risk_assessment?.confidence
        ? Math.round(analysisResult.risk_assessment.confidence * 100)
        : analysisResult.confidence;

      setHistory((prev) => [
        {
          name: file.name,
          verdict: analysisResult.verdict,
          confidence: confidence,
          risk_level: analysisResult.risk_level || "N/A",
          time: new Date().toLocaleTimeString(),
        },
        ...prev,
      ]);
    } catch (error) {
      alert(
        error.message === "Failed to fetch"
          ? "Cannot connect to the backend server. Make sure it is running on http://localhost:8000"
          : `Analysis failed: ${error.message}`
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setResult(null);
    setAnalyzing(false);
  };

  // LOGIN PAGE
  if (!loggedIn) {
    return (
      <div className="login-page">

        <div className="login-brand">
          <div className="brand-icon">🛡️</div>
          <h1>VoiceShield<span>-AI</span></h1>
          <p>AI-Powered Voice Deepfake Detection</p>
        </div>

        <form className="login-card" onSubmit={handleLogin}>

          <div className="login-header">
            <h2>Welcome Back</h2>
            <p>Login to your VoiceShield dashboard</p>
          </div>

          <label>Email Address</label>

          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <label>Password</label>

          <input
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <div className="login-options">
            <label className="remember">
              <input type="checkbox" />
              Remember me
            </label>

            <a href="#forgot">Forgot password?</a>
          </div>

          <button className="login-button">
            🔐 Login to Dashboard
          </button>

          <div className="demo-login">
            <p>Demo Mode</p>
            <small>
              Enter any email and password to continue.
            </small>
          </div>

        </form>

        <button
          className="theme-login"
          onClick={() => setDarkMode(!darkMode)}
        >
          {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
        </button>

      </div>
    );
  }

  // DASHBOARD
  return (
    <div className="app">

      {/* NAVBAR */}
      <nav className="navbar">

        <div className="logo">
          <div className="logo-shield">🛡️</div>

          <div>
            <strong>VoiceShield<span>-AI</span></strong>
            <small>DEEPFAKE DETECTION</small>
          </div>
        </div>

        <div className="nav-links">
          <a href="#dashboard">Dashboard</a>
          <a href="#history">History</a>
          <a href="#about">About</a>

          <button
            className="theme-button"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? "☀️" : "🌙"}
          </button>

          <button
            className="logout-button"
            onClick={handleLogout}
          >
            Logout
          </button>
        </div>

      </nav>

      {/* DASHBOARD HERO */}
      <main className="dashboard" id="dashboard">

        <div className="welcome">
          <div>
            <p className="eyebrow">SECURITY DASHBOARD</p>

            <h1>
              Detect Voice <span>Deepfakes</span>
            </h1>

            <p>
              Upload a voice recording and let VoiceShield-AI
              analyze its authenticity.
            </p>
          </div>

          <div className="status">
            <span></span>
            System Online
          </div>
        </div>

        {/* MAIN GRID */}
        <div className="main-grid">

          {/* UPLOAD */}
          <div className="upload-card">

            <div className="card-title">
              <div className="title-icon">🎙️</div>

              <div>
                <h2>Voice Analysis</h2>
                <p>Upload an audio file for detection</p>
              </div>
            </div>

            <label className="drop-zone">

              <div className="upload-big-icon">
                📤
              </div>

              <h3>
                {file
                  ? file.name
                  : "Drop your audio file here"}
              </h3>

              <p>
                or click to browse from your device
              </p>

              <span className="browse-button">
                Choose Audio
              </span>

              <input
                type="file"
                accept="audio/*,video/*"
                onChange={handleFileChange}
                hidden
              />

            </label>

            <div className="file-info">
              <span>✓ MP3</span>
              <span>✓ WAV</span>
              <span>✓ M4A</span>
              <span>✓ MP4</span>
            </div>

            <button
              className="analyze-button"
              disabled={!file || analyzing}
              onClick={handleAnalyze}
            >
              {analyzing
                ? "⏳ Analyzing Voice..."
                : "🔍 Analyze Voice"}
            </button>

          </div>

          {/* RESULT */}
          <div className="result-panel">

            {!analyzing && !result && (
              <div className="empty-result">

                <div className="empty-icon">
                  🧠
                </div>

                <h2>Analysis Result</h2>

                <p>
                  Your detection result will appear here
                  after analysis.
                </p>

              </div>
            )}

            {analyzing && (
              <div className="empty-result">

                <div className="loader"></div>

                <h2>Analyzing Voice...</h2>

                <p>
                  Examining acoustic and speech patterns
                </p>

                <div className="scan-line"></div>

              </div>
            )}

            {result && !analyzing && (
              <div className="result-content">

                <div className="result-top">
                  <span>DETECTION RESULT</span>

                  <div className="real-time">
                    ● Analysis Complete
                  </div>
                </div>

                <div className="verdict-box">

                  <div className="robot">
                    🤖
                  </div>

                  <div>
                    <p>VERDICT</p>
                    <h2>{result.verdict}</h2>
                  </div>

                </div>

                <div className="confidence">

                  <div className="confidence-header">
                    <span>Confidence Score</span>
                    <strong>{result.confidence}%</strong>
                  </div>

                  <div className="progress-bar">
                    <div
                      className="progress"
                      style={{
                        width: `${result.confidence}%`,
                      }}
                    ></div>
                  </div>

                </div>

                <div className="risk-engine-details">
                  <h3>📊 Risk Assessment</h3>
                  <p>Risk Level: <strong>{result?.risk_level || "N/A"}</strong></p>
                  <p>Spoof Ratio: {result?.detection?.spoof_ratio !== undefined ? (result.detection.spoof_ratio * 100).toFixed(1) + "%" : "N/A"}</p>
                  <div className="strategy-grid">
                    <div className="strategy-card">
                      <p>{result?.risk_assessment?.strategy_1?.name || "S1"}</p>
                      <strong>{result?.risk_assessment?.strategy_1?.prediction || "N/A"}</strong>
                    </div>
                    <div className="strategy-card">
                      <p>{result?.risk_assessment?.strategy_2?.name || "S2"}</p>
                      <strong>{result?.risk_assessment?.strategy_2?.prediction || "N/A"}</strong>
                    </div>
                    <div className="strategy-card">
                      <p>{result?.risk_assessment?.strategy_3?.name || "S3"}</p>
                      <strong>{result?.risk_assessment?.strategy_3?.prediction || "N/A"}</strong>
                    </div>
                  </div>
                </div>

                <div className="explanation">

                  <h3>🧠 AI Analysis</h3>

                  <p>
                    {result.explanation}
                  </p>

                </div>

                <button
                  className="reset-button"
                  onClick={resetAnalysis}
                >
                  ↻ Analyze Another File
                </button>

              </div>
            )}

          </div>

        </div>

        {/* FEATURES */}
        <section className="features">

          <div className="feature-card">
            <div>🔒</div>
            <h3>Secure Analysis</h3>
            <p>
              Your uploaded audio is processed securely.
            </p>
          </div>

          <div className="feature-card">
            <div>🧠</div>
            <h3>AI Detection</h3>
            <p>
              Advanced AI analyzes voice characteristics.
            </p>
          </div>

          <div className="feature-card">
            <div>📊</div>
            <h3>Confidence Score</h3>
            <p>
              Get an easy-to-understand detection score.
            </p>
          </div>

          <div className="feature-card">
            <div>⚡</div>
            <h3>Fast Results</h3>
            <p>
              Get analysis results within seconds.
            </p>
          </div>

        </section>

        {/* HISTORY */}
        <section className="history-section" id="history">

          <div className="section-heading">
            <div>
              <p className="eyebrow">RECENT ACTIVITY</p>
              <h2>Analysis History</h2>
            </div>
            {history.length > 0 && (
              <button className="reset-button" style={{ width: 'auto', marginTop: 0 }} onClick={handleClearHistory}>
                Clear History
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="no-history">
              No analysis performed yet.
            </div>
          ) : (
            <div className="history-list">

              {history.map((item, index) => (
                <div className="history-item" key={index}>

                  <div className="history-file">
                    🎙️
                    <div>
                      <strong>{item.name}</strong>
                      <small>{item.time}</small>
                    </div>
                  </div>

                  <span className={`history-verdict ${item.verdict}`}>
                    {item.verdict}
                  </span>

                  <span className="history-risk">
                    {item.risk_level}
                  </span>

                  <strong>
                    {item.confidence}%
                  </strong>

                </div>
              ))}

            </div>
          )}

        </section>

      </main>

      <footer id="about">
        <div>
          <strong>🛡️ VoiceShield-AI</strong>
          <p>
            AI-powered voice deepfake detection platform.
          </p>
        </div>

        <p>© 2026 VoiceShield-AI</p>
      </footer>

    </div>
  );
}

export default App;