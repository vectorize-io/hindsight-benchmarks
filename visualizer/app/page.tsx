import type { Metadata } from "next";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { AMB_RESULTS, AMB_SITE } from "@/lib/amb";
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
          text="#1 on every Agent Memory Benchmark dataset, the same datasets other memory systems run."
          href="/benchmarks"
          cta="See all results"
        >
          <div className="divide-y divide-border border-y border-border">
            {AMB_RESULTS.map((r) => (
              <Row
                key={`${r.benchmark}-${r.split}`}
                href={r.href}
                name={r.benchmark}
                detail={r.split}
                value={`${r.score}%`}
              />
            ))}
          </div>
          <a
            href={AMB_SITE}
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-[hsl(var(--primary))]"
          >
            Compare all systems{" "}
            <ArrowRight className="w-3 h-3" />
          </a>
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
