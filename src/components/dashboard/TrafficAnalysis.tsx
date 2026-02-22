"use client";

import { motion } from "framer-motion";
import { Gauge, Route, Zap, Sun } from "lucide-react";

interface TrafficData {
    avgLanes: number;
    maxLanes: number;
    speedLimits: string[];
    surfaces: string[];
    litPercentage: number;
}

interface TrafficAnalysisProps {
    data: TrafficData;
    delay?: number;
}

function Pill({ children }: { children: React.ReactNode }) {
    return (
        <span
            className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-secondary text-foreground border border-border"
        >
            {children}
        </span>
    );
}

export function TrafficAnalysis({ data, delay = 0 }: TrafficAnalysisProps) {
    const items = [
        {
            icon: <Route className="w-3.5 h-3.5" />,
            label: "Avg Lanes",
            value: data.avgLanes.toFixed(1),
            sub: `max ${data.maxLanes}`,
        },
        {
            icon: <Gauge className="w-3.5 h-3.5" />,
            label: "Speed",
            value: data.speedLimits[0] || "N/A",
            sub: data.speedLimits.length > 1 ? `+${data.speedLimits.length - 1} zones` : "",
        },
        {
            icon: <Zap className="w-3.5 h-3.5" />,
            label: "Surface",
            value: data.surfaces[0] || "N/A",
            sub: "",
        },
        {
            icon: <Sun className="w-3.5 h-3.5" />,
            label: "Lit",
            value: `${data.litPercentage}%`,
            sub: "of segments",
        },
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="glass-panel rounded-xl p-5 space-y-4"
        >
            <h3 className="text-sm font-semibold tracking-tight flex items-center gap-2">
                <Route className="w-4 h-4 text-muted-foreground" />
                Traffic & Infrastructure
            </h3>

            <div className="grid grid-cols-2 gap-3">
                {items.map((item, i) => (
                    <motion.div
                        key={item.label}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: delay + 0.2 + i * 0.1, duration: 0.4 }}
                        className="flex items-start gap-2.5 p-2.5 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors"
                    >
                        <div className="w-7 h-7 rounded-md flex items-center justify-center shrink-0 mt-0.5 bg-secondary text-muted-foreground">
                            {item.icon}
                        </div>
                        <div className="min-w-0">
                            <div className="text-sm font-bold leading-tight">{item.value}</div>
                            <div className="text-[10px] text-muted-foreground uppercase tracking-wider leading-tight">
                                {item.label}
                            </div>
                            {item.sub && (
                                <div className="text-[9px] text-muted-foreground/60 mt-0.5">
                                    {item.sub}
                                </div>
                            )}
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Surface pills */}
            {data.surfaces.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-1">
                    {data.surfaces.map((s) => (
                        <Pill key={s}>
                            {s}
                        </Pill>
                    ))}
                </div>
            )}
        </motion.div>
    );
}
