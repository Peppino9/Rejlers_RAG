import { useEffect, useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const starterPrompts = ['Sammanfatta tillståndsläge', 'Påverkan på närboende', 'Buller och vibrationer']

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
      const response = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, mode: 'citizen', project }),
      })
      if (!response.ok) throw new Error('Anrop till backend misslyckades.')
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
            Använd ALMA för att ställa frågor om projektunderlag, få sammanfattningar och hitta relevanta källor snabbt.
          </div>
        </div>
      </header>

      <section className="hero">
        <p className="eyebrow">AI Assistent</p>
        <h1>
          Du frågar, <span className="brand-inline">ALMA</span> svarar.
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
              <option value="ostlanken">Ostlänken</option>
              <option value="vastlanken">Västlänken</option>
            </select>
          </div>
          <div className="prompt-input-wrap">
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
              placeholder="Ask a question..."
              rows={1}
            />
            <button
              className="send-btn visible"
              onClick={() => askQuestion(query)}
              disabled={loading || !query.trim()}
              type="button"
              aria-label="Skicka fråga"
            >
              {loading ? <span className="send-loading" aria-label="Laddar svar" /> : <span aria-hidden="true">→</span>}
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
              <h2 className="response-title">ALMA-svar</h2>
              {hasResponse ? (
                <>
                  <p className="response-copy unfurl">
                    {active.answer.split(' ').map((word, index) => (
                      <span key={`${word}-${index}`} style={{ animationDelay: `${index * 22}ms` }}>
                        {word}&nbsp;
                      </span>
                    ))}
                  </p>
                  <div className="citations">
                    {active.sources
                      ? active.sources.split(',').map((item) => (
                          <button key={item.trim()} className="citation shimmer-hover" type="button">
                            <span className="diamond" aria-hidden="true" />
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
                <p className="response-placeholder">Svar visas här när du skickar en fråga.</p>
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
          <span className="history-empty">
            <span aria-hidden="true">◇ </span>Inga tidigare frågor ännu.
          </span>
        )}
      </section>
    </div>
  )
}

export default App
