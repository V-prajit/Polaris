"use client";

import { motion } from "framer-motion";

interface StallItem {
    name: string;
    count: number;
    type: "parking" | "road";
}

interface StallBreakdownChartProps {
    items: StallItem[];
    delay?: number;
}

export function StallBreakdownChart({
    items,
    delay = 0,
}: StallBreakdownChartProps) {
    const maxCount = Math.max(...items.map((i) => i.count), 1);
    const sorted = [...items].sort((a, b) => b.count - a.count);

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="glass-panel rounded-xl p-5 space-y-4"
        >
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold tracking-tight">
                    Stall Distribution
                </h3>
                <div className="flex gap-3">
                    <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                        <span className="w-2 h-2 rounded-full bg-[hsl(187,73%,46%)]" />
                        Parking
                    </span>
                    <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                        <span className="w-2 h-2 rounded-full bg-[hsl(38,92%,50%)]" />
                        Road
                    </span>
                </div>
            </div>

            <div className="space-y-2.5">
                {sorted.map((item, i) => {
                    const pct = (item.count / maxCount) * 100;
                    const barColor =
                        item.type === "parking"
                            ? "hsl(187, 73%, 46%)"
                            : "hsl(38, 92%, 50%)";

                    return (
                        <div key={item.name + i} className="group">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-muted-foreground truncate max-w-[180px] group-hover:text-foreground transition-colors">
                                    {item.name}
                                </span>
                                <span
                                    className="text-xs font-bold tabular-nums"
                                    style={{ color: barColor }}
                                >
                                    {item.count}
                                </span>
                            </div>
                            <div className="h-1.5 bg-secondary/50 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full rounded-full"
                                    style={{
                                        background: `linear-gradient(90deg, ${barColor}, ${barColor}88)`,
                                        boxShadow: `0 0 8px ${barColor}40`,
                                    }}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${pct}%` }}
                                    transition={{
                                        delay: delay + 0.3 + i * 0.08,
                                        duration: 0.8,
                                        ease: [0.16, 1, 0.3, 1],
                                    }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
        </motion.div>
    );
}
