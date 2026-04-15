import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').trim()
const RESOLVED_API_BASE = API_BASE || (import.meta.env.DEV ? 'http://localhost:8000' : '')

const starterPrompts = ['Sammanfatta tillståndsläge', 'Påverkan på närboende', 'Buller och vibrationer']

function SendArrowIcon() {
  return (
    <svg className="send-arrow" width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M5 12h14m-5-5 5 5-5 5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function metricLabel(key) {
  if (key === 'lix') return 'LIX'
  if (key === 'faithfulness') return 'RAGAS Trogenhet'
  if (key === 'answer_relevancy') return 'RAGAS Relevans'
  return key
}

function App() {
  const sceneRef = useRef(null)
  const textareaRef = useRef(null)
  const scenePointerRef = useRef({ x: 50, y: 50 })
  const sceneFrameRef = useRef(null)
  const [query, setQuery] = useState('')
  const [project, setProject] = useState('alla-projekt')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [active, setActive] = useState({
    question: '',
    answer: 'Ställ en fråga för att komma igång.',
    sources: '',
    metrics: null,
    timestamp: null,
  })

  const hasHistory = history.length > 0
  const hasResponse = Boolean(active.question || active.timestamp)
  const metrics = active.metrics ?? {}
  const hasMetrics = Object.keys(metrics).length > 0

  const metricPills = useMemo(
    () =>
      Object.entries(metrics).map(([k, v]) => (
        <div key={k} className="metric-pill">
          {metricLabel(k)}: {v === null || v === undefined ? '--' : Number(v).toFixed(k === 'lix' ? 2 : 3)}
        </div>
      )),
    [metrics],
  )

  useEffect(() => {
    return () => {
      if (sceneFrameRef.current) cancelAnimationFrame(sceneFrameRef.current)
    }
  }, [])

  useEffect(() => {
    if (!textareaRef.current) return
    textareaRef.current.style.height = 'auto'
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 130)}px`
  }, [query])

  function updateScenePosition() {
    if (!sceneRef.current) return
    const { x, y } = scenePointerRef.current
    sceneRef.current.style.setProperty('--bg-x', `${x}%`)
    sceneRef.current.style.setProperty('--bg-y', `${y}%`)
    sceneRef.current.style.setProperty('--bg-shift-x', `${(x - 50) * 0.18}px`)
    sceneRef.current.style.setProperty('--bg-shift-y', `${(y - 50) * 0.14}px`)
    sceneFrameRef.current = null
  }

  function queueScenePosition(x, y) {
    scenePointerRef.current = { x, y }
    if (sceneFrameRef.current) return
    sceneFrameRef.current = requestAnimationFrame(updateScenePosition)
  }

  function handleSceneMouseMove(event) {
    if (!sceneRef.current) return
    const rect = sceneRef.current.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 100
    const y = ((event.clientY - rect.top) / rect.height) * 100
    const safeX = Math.min(100, Math.max(0, x))
    const safeY = Math.min(100, Math.max(0, y))
    queueScenePosition(safeX, safeY)
  }

  function handleSceneMouseLeave() {
    queueScenePosition(50, 50)
  }

  async function askQuestion(nextQuestion) {
    const trimmed = nextQuestion.trim()
    if (!trimmed || loading) return
    setLoading(true)
    try {
      if (!RESOLVED_API_BASE) {
        throw new Error(
          'API-konfiguration saknas i frontend (VITE_API_BASE_URL). Kontakta administratör.',
        )
      }
      const response = await fetch(`${RESOLVED_API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, mode: 'citizen', project }),
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Backend svarade med ${response.status}`)
      }
      const data = await response.json()
      const entry = {
        question: trimmed,
        answer: data.answer ?? 'Inget svar returnerades.',
        sources: data.sources ?? '',
        metrics: data.metrics ?? null,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setActive(entry)
      setHistory((prev) => [entry, ...prev].slice(0, 30))
      setQuery('')
    } catch (error) {
      setActive((prev) => ({
        ...prev,
        question: trimmed,
        answer: `Förfrågan misslyckades: ${error.message}`,
      }))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="scene" ref={sceneRef} onMouseMove={handleSceneMouseMove} onMouseLeave={handleSceneMouseLeave}>
      <header className="topbar">
        <span className="brand-mark">ALMA</span>
        <div className="help-popover">
          <button className="help-trigger" type="button" aria-label="Vad kan jag använda ALMA till?">
            ?
          </button>
          <div className="help-content" role="note">
            <p className="help-content-heading">Vad kan jag använda ALMA till?</p>
            <p>
              Använd ALMA för att ställa frågor om projektunderlag, få sammanfattningar och hitta källor snabbt.
            </p>
            <p>
              <strong>Välj projekt</strong> du vill fokusera på (eller <em>Alla projekt</em>). Svaren bygger{' '}
              <strong>endast på inmatade officiella dokument</strong> för valt projekt.{' '}
              <strong>Ingen information hämtas från webben</strong> eller andra externa källor.
            </p>
          </div>
        </div>
      </header>

      <section className="hero">
        <p className="eyebrow">AI ASSISTENT</p>
        <h1 className="hero-title">
          Fråga <span className="brand-inline">ALMA</span>. Få svar.
        </h1>
        <p className="subhead">Modern intelligens för infrastrukturplanering och underbyggda beslut.</p>
      </section>

      <section className="horizon-console">
        <p className="section-title">Ställ en fråga</p>
        <div className="glass-input">
          <div className="project-picker">
            <label className="project-label" htmlFor="projectSelect">
              Välj projekt
            </label>
            <select id="projectSelect" value={project} onChange={(event) => setProject(event.target.value)}>
              <option value="alla-projekt">Alla projekt</option>
              <option value="lund-stambana">Lund Stambana</option>
            </select>
          </div>
          <div className="prompt-field">
            <textarea
              ref={textareaRef}
              id="promptInput"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onInput={(event) => {
                event.currentTarget.style.height = 'auto'
                event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 130)}px`
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  askQuestion(query)
                }
              }}
              placeholder="Ställ en fråga..."
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => askQuestion(query)}
              disabled={loading}
              type="button"
              aria-label={loading ? 'Laddar svar' : 'Skicka'}
              title="Skicka"
            >
              {loading ? <span className="send-loading" aria-hidden="true" /> : <SendArrowIcon />}
            </button>
          </div>
          <div className="prompt-actions">
            {starterPrompts.map((prompt) => (
              <button key={prompt} className="nav-link shimmer-hover" onClick={() => askQuestion(prompt)} type="button">
                {prompt}
              </button>
            ))}
          </div>
          <div className={`answer-panel ${hasResponse ? 'visible' : ''}`}>
            <article className="response-card">
              {hasResponse ? (
                <>
                  {active.question ? (
                    <div className="response-question">
                      <span className="response-question-label">Fråga</span>
                      <p className="response-question-text">{active.question}</p>
                    </div>
                  ) : null}
                  <div className="response-answer-block">
                    <h2 className="response-title">ALMA:s Svar</h2>
                    <p className="response-copy unfurl">
                      {active.answer.split(' ').map((word, index) => (
                        <span key={`${word}-${index}`} style={{ animationDelay: `${index * 22}ms` }}>
                          {word}&nbsp;
                        </span>
                      ))}
                    </p>
                  </div>
                  <div className="citations">
                    {active.sources
                      ? active.sources.split(',').map((item) => (
                          <button key={item.trim()} className="citation shimmer-hover" type="button">
                            <span className="citation-marker" aria-hidden="true" />
                            {item.trim()}
                          </button>
                        ))
                      : null}
                    {metricPills}
                    <span className="citation">Saved {active.timestamp ?? '--'}</span>
                    {hasMetrics ? <span className="citation verified">Verifierad data</span> : null}
                  </div>
                </>
              ) : (
                <div className="response-answer-block">
                  <h2 className="response-title">ALMA:s Svar</h2>
                  <p className="response-placeholder">Svar visas här när du skickar en fråga.</p>
                </div>
              )}
            </article>
          </div>
        </div>
      </section>

      <section className="history-section">
        <div className="history-header">
          <p className="history-title">Historik</p>
          <button className="nav-link shimmer-hover" type="button" onClick={() => setHistory([])}>
            Rensa historik
          </button>
        </div>
      </section>

      <section className="history-strip">
        {hasHistory ? (
          history.slice(0, 4).map((item, idx) => (
            <button key={`${item.timestamp}-${idx}`} className="history-chip shimmer-hover" onClick={() => setActive(item)}>
              {item.timestamp} · {item.question.slice(0, 45)}
            </button>
          ))
        ) : (
          <span className="history-empty">Inga tidigare frågor ännu.</span>
        )}
      </section>
    </div>
  )
}

export default App
