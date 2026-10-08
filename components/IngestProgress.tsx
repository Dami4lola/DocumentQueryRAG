import { Check, FileText, LoaderCircle } from "lucide-react";

export type IngestState = {
  fileName: string;
  stage: "reading" | "embedding" | "indexing";
  done?: number;
  total?: number;
};

const STAGES = [
  { id: "reading", label: "Reading the document" },
  { id: "embedding", label: "Understanding the passages" },
  { id: "indexing", label: "Building the search index" },
] as const;

export function IngestProgress({ state }: { state: IngestState }) {
  const current = STAGES.findIndex((stage) => stage.id === state.stage);
  const embedded = state.total ? (state.done ?? 0) / state.total : 0;
  const width = state.stage === "reading" ? 8 : state.stage === "embedding" ? 10 + embedded * 80 : 94;

  return (
    <div className="mx-auto w-full max-w-lg px-4 py-16 sm:py-24" aria-live="polite">
      <div className="rounded-3xl border border-line bg-surface p-6 sm:p-8">
        <div className="flex items-center gap-3">
          <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-brand-soft text-brand">
            <FileText className="size-5" />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-bold tracking-[0.12em] text-muted uppercase">Preparing</p>
            <p className="truncate font-display font-bold">{state.fileName}</p>
          </div>
        </div>

        <ol className="mt-6 space-y-3">
          {STAGES.map((stage, index) => {
            const status = index < current ? "done" : index === current ? "active" : "pending";
            return (
              <li key={stage.id} className="flex items-center gap-3 text-sm">
                <span
                  className={`grid size-6 shrink-0 place-items-center rounded-full ${
                    status === "done"
                      ? "bg-brand text-white dark:text-[#0c1310]"
                      : status === "active"
                        ? "bg-brand-soft text-brand"
                        : "border border-line text-muted"
                  }`}
                >
                  {status === "done" ? (
                    <Check className="size-3.5" />
                  ) : status === "active" ? (
                    <LoaderCircle className="size-3.5 animate-spin" />
                  ) : (
                    <span className="text-[0.65rem]">{index + 1}</span>
                  )}
                </span>
                <span className={status === "pending" ? "text-muted" : "font-medium"}>{stage.label}</span>
                {stage.id === "embedding" && status === "active" && state.total ? (
                  <span className="ml-auto text-xs text-muted tabular-nums">
                    {state.done ?? 0} / {state.total}
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>

        <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-line">
          <div
            className="h-full rounded-full bg-brand transition-all duration-500"
            style={{ width: `${width}%` }}
          />
        </div>
      </div>
    </div>
  );
}
