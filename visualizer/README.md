# Benchmark Visualizer (Next.js)

A static Next.js application for visualizing LongMemEval and LoComo benchmark results, optimized for deployment on Vercel.

## Features

- 📊 Static site generation - fast, CDN-distributed
- 🎨 Beautiful UI with Tailwind CSS + shadcn/ui theme
- 🗜️ Supports both `.json` and `.json.gz` result files
- 🚀 Optimized for Vercel deployment
- 📈 Category breakdown - view accuracy by question category
- 🔍 Advanced filters - search by ID, filter by correctness and category
- 🎯 Item detail pages - dive deep into individual questions with reasoning and retrieved memories

## Prerequisites

- Node.js 18+
- npm or yarn

## Development

1. Install dependencies:
```bash
npm install
```

2. Run development server:
```bash
npm run dev
```

3. Open http://localhost:9998

## Building

Build the static site:
```bash
npm run build
```

This generates a static site in the `out/` directory.

## Deployment to Vercel

### Option 1: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd visualizer-web
vercel
```

### Option 2: Deploy via GitHub

1. Push to GitHub
2. Import project in Vercel dashboard
3. Vercel will auto-detect Next.js and deploy

## File Structure

```
visualizer-web/
├── app/
│   ├── page.tsx              # Home/benchmark selector
│   ├── longmemeval/
│   │   └── page.tsx          # LongMemEval results
│   └── locomo/
│       └── page.tsx          # LoComo results
├── lib/
│   └── data.ts               # Data loading utilities
└── results/                  # Copy your result files here
    ├── longmemeval.json.gz
    └── locomo.json.gz
```

## Important Notes

- Result files are loaded at **build time** (not runtime)
- The app reads from `../results/` relative to the build directory
- Supports both `.json` and `.json.gz` files automatically
- For Vercel deployment, compressed files (`*.json.gz`) are recommended to stay under size limits

## Cost

- **Free** on Vercel for hobby projects
- Unlimited bandwidth on free tier
- Static site = no server costs
