import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Clipboard,
  Eraser,
  FileText,
  HeartPulse,
  Loader2,
  MessageSquareQuote,
  Play,
  Sparkles,
  Wand2
} from "lucide-react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const REQUEST_TIMEOUT_MS = 120000;

const TASKS = {
  generate: {
    label: "Generate",
    endpoint: "/generate-text",
    method: "POST",
    icon: Sparkles,
    description: "Draft polished text from a prompt.",
    defaults: {
      prompt: "Write a concise product update for an AI model serving dashboard.",
      max_new_tokens: 70,
      temperature: 0.45,
      top_p: 0.85
    }
  },
  summarize: {
    label: "Summarize",
    endpoint: "/summarize",
    method: "POST",
    icon: FileText,
    description: "Condense long content into a tight summary.",
    defaults: {
      text:
        "Our team launched a new AI workspace that helps users draft content, summarize long notes, understand sentiment, and answer questions from supplied context. The experience is designed to be simple enough for daily use while still producing clear, useful results.",
      max_length: 120,
      min_length: 30
    }
  },
  sentiment: {
    label: "Sentiment",
    endpoint: "/sentiment",
    method: "POST",
    icon: HeartPulse,
    description: "Classify text as positive, negative, or neutral.",
    defaults: {
      text: "The new inference dashboard is fast, clear, and easy to operate."
    }
  },
  qa: {
    label: "Q&A",
    endpoint: "/question-answering",
    method: "POST",
    icon: MessageSquareQuote,
    description: "Ask a question against a supplied context.",
    defaults: {
      question: "Which model tasks are available?",
      context:
        "This AI workspace can generate text from prompts, summarize long passages, detect sentiment, and answer questions from a context paragraph. Users choose a task, provide content, and receive a focused result.",
      max_answer_length: 80
    }
  }
};

function App() {
  const [activeTask, setActiveTask] = useState("generate");
  const [forms, setForms] = useState(() =>
    Object.fromEntries(Object.entries(TASKS).map(([key, task]) => [key, task.defaults]))
  );
  const [health, setHealth] = useState({ state: "checking", message: "Checking service" });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const task = TASKS[activeTask];
  const form = forms[activeTask];
  const TaskIcon = task.icon;

  useEffect(() => {
    checkHealth();
  }, []);

  async function checkHealth() {
    setHealth({ state: "checking", message: "Checking service" });
    try {
      const response = await fetchApi("/health");
      if (!response.ok) {
        throw new Error(`Health check failed with ${response.status}`);
      }
      const data = await response.json();
      setHealth({
        state: "online",
        message: data.status === "healthy" ? "Service ready" : "Service available"
      });
    } catch (err) {
      setHealth({ state: "offline", message: "Service unavailable" });
    }
  }

  function updateField(field, value) {
    setForms((current) => ({
      ...current,
      [activeTask]: {
        ...current[activeTask],
        [field]: value
      }
    }));
  }

  function resetTask() {
    setForms((current) => ({
      ...current,
      [activeTask]: TASKS[activeTask].defaults
    }));
    setResult(null);
    setError("");
  }

  async function runTask(event) {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetchApi(task.endpoint, {
        method: task.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });
      const data = await readApiResponse(response);

      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "Request failed";
        throw new Error(detail);
      }

      setResult(data);
      setHistory((current) => [
        {
          id: crypto.randomUUID(),
          task: task.label,
          at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          preview: getResultPreview(data)
        },
        ...current
      ].slice(0, 5));
    } catch (err) {
      setError(getRequestError(err));
    } finally {
      setLoading(false);
    }
  }

  async function copyResult() {
    const text = result ? getResultPreview(result) : "";
    if (text) {
      await navigator.clipboard.writeText(text);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={25} />
          </div>
          <div>
            <p className="eyebrow">Model Ops</p>
            <h1>AI Serving Console</h1>
          </div>
        </div>

        <div className={`status ${health.state}`}>
          {health.state === "online" ? <CheckCircle2 size={18} /> : <Activity size={18} />}
          <span>{health.message}</span>
          <button type="button" className="icon-button" onClick={checkHealth} aria-label="Refresh health">
            <Activity size={16} />
          </button>
        </div>

        <nav className="task-nav" aria-label="AI tasks">
          {Object.entries(TASKS).map(([key, item]) => {
            const Icon = item.icon;
            return (
              <button
                key={key}
                type="button"
                className={activeTask === key ? "active" : ""}
                onClick={() => {
                  setActiveTask(key);
                  setResult(null);
                  setError("");
                }}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <section className="endpoint-list" aria-label="Workspace guidance">
          <div className="section-title">Workspace</div>
          <div className="workspace-note">
            <Wand2 size={18} />
            <p>Choose a task, enter your content, and get a clean AI response.</p>
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">AI writing and analysis</p>
            <h2>{task.label}</h2>
            <p>{task.description}</p>
          </div>
          <div className="model-chip">
            <Bot size={17} />
            <span>Smart assistant</span>
          </div>
        </header>

        <div className="content-grid">
          <form className="panel control-panel" onSubmit={runTask}>
            <div className="panel-heading">
              <TaskIcon size={20} />
              <h3>Request</h3>
            </div>

            {activeTask === "generate" && (
              <>
                <Field label="Prompt" count={`${form.prompt.length}/1000`}>
                  <textarea
                    value={form.prompt}
                    maxLength={1000}
                    onChange={(event) => updateField("prompt", event.target.value)}
                    required
                  />
                </Field>
              </>
            )}

            {activeTask === "summarize" && (
              <Field label="Source text" count={`${form.text.length}/10000`}>
                <textarea
                  value={form.text}
                  minLength={30}
                  maxLength={10000}
                  onChange={(event) => updateField("text", event.target.value)}
                  required
                />
              </Field>
            )}

            {activeTask === "sentiment" && (
              <Field label="Text" count={`${form.text.length}/3000`}>
                <textarea
                  value={form.text}
                  maxLength={3000}
                  onChange={(event) => updateField("text", event.target.value)}
                  required
                />
              </Field>
            )}

            {activeTask === "qa" && (
              <>
                <Field label="Question" count={`${form.question.length}/1000`}>
                  <input
                    value={form.question}
                    maxLength={1000}
                    onChange={(event) => updateField("question", event.target.value)}
                    required
                  />
                </Field>
                <Field label="Context" count={`${form.context.length}/10000`}>
                  <textarea
                    value={form.context}
                    minLength={20}
                    maxLength={10000}
                    onChange={(event) => updateField("context", event.target.value)}
                    required
                  />
                </Field>
                <Slider label="Max answer length" min={5} max={200} value={form.max_answer_length} onChange={(value) => updateField("max_answer_length", value)} />
              </>
            )}

            <div className="form-actions">
              <button type="button" className="secondary-button" onClick={resetTask}>
                <Eraser size={17} />
                Reset
              </button>
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                Run {task.label}
              </button>
            </div>
          </form>

          <section className="panel response-panel">
            <div className="panel-heading response-heading">
              <div>
                <Activity size={20} />
                <h3>Response</h3>
              </div>
              <button type="button" className="icon-button" onClick={copyResult} disabled={!result} aria-label="Copy response">
                <Clipboard size={17} />
              </button>
            </div>

            {error && (
              <div className="error-box">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}

            {!error && !result && (
              <div className="empty-state">
                <TaskIcon size={34} />
                <h3>Ready when you are</h3>
                <p>Enter your prompt or content and run the task to see the answer here.</p>
              </div>
            )}

            {result && <ResultView result={result} />}
          </section>
        </div>

        <section className="history-strip" aria-label="Recent requests">
          <div className="section-title">Recent work</div>
          {history.length === 0 ? (
            <p>No results yet.</p>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <article key={item.id} className="history-item">
                  <span>{item.at}</span>
                  <strong>{item.task}</strong>
                  <p>{item.preview}</p>
                </article>
              ))}
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

async function readApiResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  const text = await response.text();
  return { detail: text || `Request failed with ${response.status}` };
}

async function fetchApi(path, options = {}) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      mode: "cors",
      signal: controller.signal
    });
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function getRequestError(error) {
  if (error.name === "AbortError") {
    return `The API request timed out at ${API_BASE_URL}. The backend may still be loading a model; try again after it finishes starting.`;
  }

  if (error instanceof TypeError) {
    if (API_BASE_URL.includes("127.0.0.1") || API_BASE_URL.includes("localhost")) {
      return `Unable to reach the local API at ${API_BASE_URL}. Start the backend with: cd backend; .\\.venv\\Scripts\\python -m uvicorn app:app --host 127.0.0.1 --port 8000`;
    }

    return `Unable to reach the API at ${API_BASE_URL}. Check that the backend is deployed, awake, and allowed by CORS.`;
  }

  return error.message || "Something went wrong";
}

function Field({ label, count, children }) {
  return (
    <label className="field">
      <span>
        {label}
        {count && <small>{count}</small>}
      </span>
      {children}
    </label>
  );
}

function Slider({ label, min, max, step = 1, value, onChange }) {
  return (
    <label className="slider-field">
      <span>
        {label}
        <output>{value}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ResultView({ result }) {
  if (result.task === "sentiment-analysis") {
    const confidence = Math.round(result.score * 100);
    return (
      <div className="result-stack">
        <div className={`sentiment-badge ${result.sentiment}`}>{result.sentiment}</div>
        <Metric label="Detected tone" value={friendlyLabel(result.sentiment)} featured />
        <div className="confidence">
          <span>Confidence</span>
          <strong>{confidence}%</strong>
          <div>
            <i style={{ width: `${confidence}%` }} />
          </div>
        </div>
      </div>
    );
  }

  if (result.task === "question-answering") {
    return (
      <div className="result-stack">
        <Metric label="Answer" value={result.answer || "No answer returned"} featured />
        <Metric label="Confidence" value={`${Math.round(result.score * 100)}%`} />
      </div>
    );
  }

  if (result.task === "summarization") {
    return (
      <div className="result-stack">
        <Metric label="Summary" value={result.summary} featured />
      </div>
    );
  }

  return (
    <div className="result-stack">
      <Metric label="Generated text" value={result.generated_text} featured />
    </div>
  );
}

function Metric({ label, value, featured = false }) {
  return (
    <div className={featured ? "metric featured" : "metric"}>
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

function getResultPreview(data) {
  return data.generated_text || data.summary || data.answer || `${data.sentiment} (${Math.round(data.score * 100)}%)`;
}

function friendlyLabel(value) {
  if (value === "positive") return "Positive and encouraging";
  if (value === "negative") return "Negative or critical";
  return "Neutral or mixed";
}

export default App;
