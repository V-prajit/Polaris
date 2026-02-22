"use client";

import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

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
            className="glass-panel rounded-xl p-5 space-y-3"
        >
            <h3 className="text-sm font-semibold tracking-tight flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-muted-foreground" />
                {title}
            </h3>

            <p className="text-xs text-muted-foreground leading-relaxed">
                {text}
            </p>

            {/* Divider */}
            <div className="h-px w-full bg-border" />
        </motion.div>
    );
}
