"use client";

import { motion } from "framer-motion";
import { BrainCircuit, Sparkles } from "lucide-react";

interface InsightCardProps {
    title?: string;
    text: string;
    model?: string;
    delay?: number;
}

export function InsightCard({
    title = "AI Analysis",
    text,
    model = "SegFormer-B4",
    delay = 0,
}: InsightCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="relative rounded-xl overflow-hidden"
        >
            {/* Gradient border effect */}
            <div
                className="absolute inset-0 rounded-xl"
                style={{
                    background:
                        "linear-gradient(135deg, hsla(187, 73%, 46%, 0.3), hsla(213, 74%, 47%, 0.2), hsla(187, 73%, 46%, 0.1))",
                    padding: "1px",
                }}
            />
            <div className="relative glass-panel rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold tracking-tight flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-primary" />
                        {title}
                    </h3>
                    <span className="flex items-center gap-1.5 text-[9px] font-mono text-muted-foreground/60 bg-secondary/40 px-2 py-0.5 rounded-full">
                        <BrainCircuit className="w-3 h-3" />
                        {model}
                    </span>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                    {text}
                </p>

                {/* Shimmer strip */}
                <div className="h-px w-full shimmer-bg rounded-full" />
            </div>
        </motion.div>
    );
}
