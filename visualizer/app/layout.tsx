import Script from 'next/script'
import type { Metadata } from 'next'
import { Space_Grotesk, Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  weight: ['700'],
})

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  weight: ['400', '500', '600'],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  weight: ['400', '500'],
})

const siteUrl = 'https://benchmarks.hindsight.vectorize.io'

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: {
    default: 'Hindsight Benchmarks — #1 on Agent Memory Benchmark',
    template: '%s | Hindsight Benchmarks',
  },
  description:
    'Hindsight leads every dataset on the Agent Memory Benchmark. Explore results across LongMemEval, LoComo, BEAM, LifeBench, and PersonaMem — plus model leaderboards for LLMs, rerankers, and embeddings.',
  icons: {
    icon: '/favicon.png',
  },
  openGraph: {
    type: 'website',
    siteName: 'Hindsight Benchmarks',
    title: 'Hindsight Benchmarks — #1 on Agent Memory Benchmark',
    description:
      'Hindsight leads every dataset on the Agent Memory Benchmark. Explore benchmark results and model leaderboards for LLMs, rerankers, and embeddings.',
    url: siteUrl,
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Hindsight Benchmarks — #1 on Agent Memory Benchmark',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Hindsight Benchmarks — #1 on Agent Memory Benchmark',
    description:
      'Hindsight leads every dataset on the Agent Memory Benchmark. Explore benchmark results and model leaderboards.',
    images: ['/og-image.png'],
  },
  alternates: {
    canonical: siteUrl,
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <Script
          defer
          src="https://analytics.hindsight.vectorize.io/script.js"
          data-website-id="e59d3201-3a11-46e3-b48f-895f12d148fb"
          strategy="afterInteractive"
        />
      </head>
      <body className="antialiased font-body min-h-screen bg-background text-foreground">
        {children}
      </body>
    </html>
  )
}
