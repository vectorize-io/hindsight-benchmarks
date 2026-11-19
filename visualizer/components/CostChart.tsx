import type { Cost } from '@/lib/benchmarks'

// A card's score against what each setup costs: one dot per system, its label in a column right of the
// plot joined by a line, so close dots never print over each other. Up and to the left is better.
const pct = (v: number) => (v * 100).toFixed(1)

// Round tick steps: 1, 2 or 5 times a power of ten, about `n` of them up to `max`.
function ticks(max: number, n = 4) {
  const raw = max / n
  const pow = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw)!
  return Array.from({ length: Math.ceil(max / step) + 1 }, (_, i) => +(i * step).toPrecision(6))
}

export function CostChart({ cost }: { cost: Cost }) {
  const { dots } = cost
  const W = 560, H = 240, L = 34, R = 190, T = 14, B = 36
  const vals = dots.map((d) => d.value * 100)
  const span = Math.max(...vals) - Math.min(...vals)
  const yStep = span > 40 ? 20 : span > 15 ? 10 : 5
  const lo = Math.floor(Math.min(...vals) / yStep) * yStep, hi = Math.ceil(Math.max(...vals) / yStep) * yStep
  const xTicks = ticks(Math.max(...dots.map((d) => d.cost)))
  const x = (c: number) => L + (c / xTicks[xTicks.length - 1]) * (W - L - R)
  const y = (v: number) => T + (1 - (v * 100 - lo) / (hi - lo || 1)) * (H - T - B)
  const yTicks = Array.from({ length: (hi - lo) / yStep + 1 }, (_, i) => lo + i * yStep)
  // Labels top to bottom in score order, each pushed down below the one above it.
  let last = -Infinity
  const labels = [...dots].sort((a, b) => b.value - a.value).map((d) => ({ d, ly: (last = Math.max(y(d.value), last + 13)) }))
  return (
    <div>
      <h4 className="text-xs font-medium">Score against cost</h4>
      <p className="text-xs text-muted-foreground mb-1">{cost.about}</p>
      <svg viewBox={`0 0 ${W} ${Math.max(H, last + 8)}`} className="w-full h-auto" role="img" aria-label={`Score against ${cost.axis}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t / 100)} y2={y(t / 100)} stroke="hsl(var(--border))" />
            <text x={L - 6} y={y(t / 100)} dy="0.32em" textAnchor="end" fontSize="10" fill="hsl(var(--muted-foreground))">{t}</text>
          </g>
        ))}
        {xTicks.map((c) => (
          <text key={c} x={x(c)} y={H - B + 16} textAnchor="middle" fontSize="10" fill="hsl(var(--muted-foreground))">{cost.money(c)}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="hsl(var(--muted-foreground))">{cost.axis} →</text>
        {labels.map(({ d, ly }) => (
          <a key={d.label} href={d.href}>
            <title>{`${d.label}: ${pct(d.value)} at ${cost.money(d.cost)}`}</title>
            <line x1={x(d.cost) + 6} y1={y(d.value)} x2={W - R + 10} y2={ly} stroke="hsl(var(--muted-foreground))" strokeOpacity="0.4" />
            <circle cx={x(d.cost)} cy={y(d.value)} r={d.ours ? 6 : 5} fill={d.ours ? '#0074d9' : 'hsl(var(--muted-foreground))'} />
            <text
              x={W - R + 14}
              y={ly}
              dy="0.32em"
              fontSize="11"
              fontWeight={d.ours ? 700 : 400}
              fill={d.ours ? 'hsl(var(--foreground))' : 'hsl(var(--muted-foreground))'}
            >
              {d.label} {pct(d.value)}
            </text>
          </a>
        ))}
      </svg>
    </div>
  )
}
