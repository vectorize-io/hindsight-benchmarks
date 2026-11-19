import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { SECTIONS, rank } from "@/lib/benchmarks";
import { BOARDS } from "@/lib/data";
import { FAMILIES } from "@/lib/models";
import { PageHero } from "@/components/Page";
import { ModelScatter } from "@/components/BoardView";

export const metadata: Metadata = {
  title: { absolute: "Hindsight Benchmarks" },
  description:
    "How Hindsight scores as a whole on the Agent Memory Benchmark, and which models to run inside it.",
};

// One column per place to go next, side by side so all three show without scrolling.
function Column({
  label,
  title,
  text,
  href,
  cta,
  children,
}: {
  label: string;
  title: React.ReactNode;
  text: string;
  href: string;
  cta: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col min-w-0">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
        {label}
      </div>
      <h2 className="font-heading font-extrabold text-xl tracking-[-0.02em] leading-tight mb-2">
        {title}
      </h2>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        {text}
      </p>
      <div className="mb-4">{children}</div>
      <Link
        href={href}
        className="mt-auto inline-flex items-center justify-center gap-1.5 h-9 px-4 rounded-lg font-heading font-bold text-sm text-white hover:brightness-110 transition"
        style={{
          background: "linear-gradient(90deg, #0074d9 0%, #009296 100%)",
        }}
      >
        {cta} <ArrowRight className="w-4 h-4" />
      </Link>
    </section>
  );
}

function Row({
  href,
  name,
  detail,
  value,
}: {
  href: string;
  name: string;
  detail?: string;
  value?: string;
}) {
  const cls =
    "group flex items-center gap-3 py-1.5 text-sm hover:text-[hsl(var(--primary))] transition-colors";
  const body = (
    <>
      <span className="font-medium truncate">{name}</span>
      {detail && (
        <span className="font-mono text-xs text-muted-foreground truncate">
          {detail}
        </span>
      )}
      <span className="ml-auto font-mono font-semibold tabular-nums">
        {value}
      </span>
    </>
  );
  return href.startsWith("http") ? (
    <a href={href} className={cls}>
      {body}
    </a>
  ) : (
    <Link href={href} className={cls}>
      {body}
    </Link>
  );
}

// Every benchmark where Hindsight is 1st, from the same data as /benchmarks. A card with several
// panels (BEAM's sizes) counts when Hindsight is 1st on all of them; the rest are left out.
function firsts() {
  return SECTIONS.flatMap((s) =>
    s.cards
      .filter((c) => c.panels.every((p) => rank(p)?.place === 1))
      .map((c) => ({ c, href: `/benchmarks#${s.key}` })),
  );
}

export default function Home() {
  const allOps = BOARDS.find((b) => b.id === "all-operations")!;
  return (
    <>
      <PageHero
        title="Hindsight Benchmarks"
        subtitle="How Hindsight scores as a whole on public memory benchmarks, and which models to run inside it."
      />
      <main className="max-w-6xl mx-auto px-4 pt-8 grid md:grid-cols-[1fr_1.4fr_1fr] gap-10 items-stretch">
        <Column
          label="Benchmarks"
          title={
            <>
              How Hindsight compares to{" "}
              <span className="gradient-primary-text">
                other memory systems
              </span>
            </>
          }
          text="State of the art on these Agent Memory Benchmark datasets, against every system that publishes a score."
          href="/benchmarks"
          cta="See all results"
        >
          <div className="divide-y divide-border border-y border-border">
            {firsts().map(({ c, href }) => (
              <Link key={c.key} href={href} className="group flex items-center gap-3 py-2 hover:text-[hsl(var(--primary))] transition-colors">
                <span
                  className="shrink-0 text-[10px] font-bold text-white px-1.5 py-0.5 rounded-full"
                  style={{ background: "linear-gradient(90deg, #0074d9 0%, #009296 100%)" }}
                >
                  #1
                </span>
                <span className="min-w-0">
                  <span className="block font-medium text-sm truncate">{c.bench}</span>
                  <span className="block text-xs text-muted-foreground truncate">{c.title}</span>
                </span>
              </Link>
            ))}
          </div>
        </Column>

        <Column
          label="Model leaderboards"
          title="Which model to run, operation by operation"
          text="One leaderboard per Hindsight operation. Each swaps one model and keeps the rest fixed."
          href="/leaderboards"
          cta="All leaderboards"
        >
          <ModelScatter rows={allOps.tabs[0].rows} task={allOps.title} bare />
        </Column>

        <Column
          label="Hindsight Models"
          title="Our own models, on public benchmarks"
          text="The models behind Hindsight, next to the ones people use today, on benchmarks we did not make."
          href="/models"
          cta="All models"
        >
          <div className="divide-y divide-border border-y border-border">
            {FAMILIES.map((f) => {
              const b = f.benchmarks[0];
              const score = b && f.rows[0]?.scores[b.id];
              return (
                <Row
                  key={f.id}
                  href={`/models/${f.id}`}
                  name={f.title}
                  detail={f.model}
                  value={score != null ? score.toFixed(1) : undefined}
                />
              );
            })}
          </div>
        </Column>
      </main>
    </>
  );
}
