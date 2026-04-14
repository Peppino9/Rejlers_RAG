/**
 * Full-screen interactive dot grid (canvas under #app).
 * Skips animation when prefers-reduced-motion is set.
 */
export function initInteractiveDotField() {
  const canvas = document.getElementById('dotCanvas')
  if (!canvas) return

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (reduceMotion) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const GRID = 24
  const BASE_DOT = 1.15
  const PROXIMITY = 160
  const DRIFT_SPEED = 0.000011
  let width = 0
  let height = 0
  let dpr = 1
  let mouseX = -9999
  let mouseY = -9999
  let rafId = 0

  function resizeCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2)
    width = window.innerWidth
    height = window.innerHeight
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  function drawDots(ts) {
    ctx.clearRect(0, 0, width, height)
    const driftX = ((ts * DRIFT_SPEED * 18) % GRID + GRID) % GRID
    const driftY = ((ts * DRIFT_SPEED * 11) % GRID + GRID) % GRID
    for (let x = -GRID; x < width + GRID; x += GRID) {
      for (let y = -GRID; y < height + GRID; y += GRID) {
        const px = x + driftX
        const py = y + driftY
        const dist = Math.hypot(px - mouseX, py - mouseY)
        let t = 0
        if (dist < PROXIMITY) {
          t = 1 - dist / PROXIMITY
          t *= t
        }
        const radius = BASE_DOT * (1 + t * 2.2)
        const base = 26 + t * 229
        const alpha = 0.1 + t * 0.92
        ctx.fillStyle = `rgba(${base}, ${base}, ${Math.min(255, base + 6)}, ${alpha})`
        ctx.beginPath()
        ctx.arc(px, py, radius, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    rafId = requestAnimationFrame(drawDots)
  }

  function onResize() {
    resizeCanvas()
  }

  function onMouseMove(e) {
    mouseX = e.clientX
    mouseY = e.clientY
  }

  window.addEventListener('resize', onResize)
  window.addEventListener('mousemove', onMouseMove)
  resizeCanvas()
  rafId = requestAnimationFrame(drawDots)

  return () => {
    window.removeEventListener('resize', onResize)
    window.removeEventListener('mousemove', onMouseMove)
    cancelAnimationFrame(rafId)
  }
}
