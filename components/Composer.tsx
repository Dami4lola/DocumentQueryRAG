"use client";

import { useRef, useState } from "react";
import { ArrowUp } from "lucide-react";

type Props = {
  disabled: boolean;
  placeholder: string;
  onSend: (question: string) => void;
};

export function Composer({ disabled, placeholder, onSend }: Props) {
  const [value, setValue] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null);

  function resize() {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }

  function submit() {
    const question = value.trim();
    if (!question || disabled) return;
    onSend(question);
    setValue("");
    requestAnimationFrame(resize);
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
      className="flex items-end gap-2 rounded-2xl border border-line bg-surface p-2 shadow-sm transition focus-within:border-brand focus-within:ring-2 focus-within:ring-brand/15"
    >
      <label htmlFor="question" className="sr-only">
        Ask a question about your document
      </label>
      <textarea
        id="question"
        ref={textarea}
        rows={1}
        value={value}
        onChange={(event) => {
          setValue(event.target.value);
          resize();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder}
        className="max-h-[200px] min-h-10 flex-1 resize-none bg-transparent px-2 py-2 outline-none placeholder:text-muted/80"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="grid size-10 shrink-0 place-items-center rounded-xl bg-brand text-white transition hover:bg-brand-hover disabled:opacity-40 dark:text-[#0c1310]"
        aria-label="Send question"
      >
        <ArrowUp className="size-5" />
      </button>
    </form>
  );
}
