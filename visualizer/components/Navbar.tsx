'use client'

import { useEffect, useState } from 'react'
import { usePathname } from 'next/navigation'
import Link from 'next/link'
import { ArrowUpRight, Moon, Sun } from 'lucide-react'

// The docs site's top bar (hindsight-docs/src/css/custom.css): logo, Space
// Grotesk links on a 4rem bar, the section you are in as gradient text on a
// tinted surface, then GitHub, the theme toggle and the one call to action.
const DOCS = 'https://hindsight.vectorize.io'

// This site's two sections, then Docs (leaves the site) after a separator.
const LINKS = [
  { label: 'Benchmarks', href: '/benchmarks', section: (p: string) => p.startsWith('/benchmarks') },
  { label: 'Model leaderboards', href: '/leaderboards', section: (p: string) => p.startsWith('/leaderboards') },
  { label: 'Docs', href: `${DOCS}/`, external: true },
]

const linkClass =
  'inline-flex items-center h-8 px-3 rounded-[0.625rem] font-heading font-bold text-sm transition-colors'

export function Navbar() {
  const pathname = usePathname() ?? ''
  return (
    <nav className="sticky top-0 z-50 h-16 border-b border-border bg-background">
      <div className="h-full px-4 flex items-center gap-2">
        {/* mb-2, as the docs' .navbar__logo: the wordmark sits low in the image, this lines it up with the links. */}
        <Link href="/" className="flex items-center mr-4 mb-2 shrink-0" aria-label="Hindsight Benchmarks home">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="Hindsight" className="h-8 w-auto" />
        </Link>
        <div className="hidden md:flex items-center gap-1">
          {LINKS.map((l) => {
            if (l.external)
              return (
                <span key={l.label} className="flex items-center gap-1">
                  <span className="w-px h-4 bg-border mx-2" aria-hidden />
                  <a href={l.href} className={`${linkClass} gap-1 text-foreground hover:bg-secondary`}>
                    {l.label}
                    <ArrowUpRight className="w-3.5 h-3.5 text-muted-foreground" aria-label="(opens the docs)" />
                  </a>
                </span>
              )
            const active = l.section?.(pathname)
            return (
              <Link
                key={l.label}
                href={l.href}
                aria-current={active ? 'page' : undefined}
                className={`${linkClass} ${active ? 'nav-active' : 'text-foreground hover:bg-secondary'}`}
              >
                {l.label}
              </Link>
            )
          })}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <a
            href="https://github.com/vectorize-io/hindsight"
            className={`${linkClass} text-foreground hover:bg-secondary gap-1.5`}
            aria-label="Hindsight on GitHub"
          >
            <svg className="w-[18px] h-[18px]" fill="currentColor" viewBox="0 0 24 24" aria-hidden>
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            <span className="hidden lg:inline font-mono text-[0.8rem] font-semibold">54.3k</span>
          </a>
          <ThemeToggle />
          <span className="hidden sm:block w-px h-5 bg-border mx-2" aria-hidden />
          <a
            href="https://ui.hindsight.vectorize.io"
            className="hidden sm:inline-flex items-center h-8 px-3.5 rounded-lg font-heading font-bold text-sm text-white hover:brightness-110 transition"
            style={{ background: 'linear-gradient(90deg, #0074d9 0%, #009296 100%)' }}
          >
            Start free
          </a>
        </div>
      </div>
    </nav>
  )
}

function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null)

  useEffect(() => {
    const set = document.documentElement.dataset.theme
    setDark(set ? set === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches)
  }, [])

  const toggle = () => {
    const next = dark ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    try {
      localStorage.setItem('v2-theme', next)
    } catch {}
    setDark(!dark)
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className={`${linkClass} text-foreground hover:bg-secondary`}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {dark ? <Sun className="w-[18px] h-[18px]" /> : <Moon className="w-[18px] h-[18px]" />}
    </button>
  )
}

// Runs before paint so a saved choice never flashes the other theme.
export const themeScript = `try{var t=localStorage.getItem('v2-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}`
