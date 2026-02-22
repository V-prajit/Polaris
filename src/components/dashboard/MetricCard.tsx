"use client";

import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect, type ReactNode } from "react";

interface MetricCardProps {
    icon: ReactNode;
    value: number;
    label: string;
    suffix?: string;
    delay?: number;
    color?: string;
}

function AnimatedNumber({
    value,
    delay,
    suffix = "",
}: {
    value: number;
    delay: number;
    suffix?: string;
}) {
    const motionVal = useMotionValue(0);
    const display = useTransform(motionVal, (v) => `${Math.round(v)}${suffix}`);

    useEffect(() => {
        const controls = animate(motionVal, value, {
            delay,
            duration: 1.2,
            ease: [0.16, 1, 0.3, 1],
        });
        return controls.stop;
    }, [motionVal, value, delay]);

    return (
        <motion.span className="text-2xl font-bold tracking-tight">
            {display}
        </motion.span>
    );
}

export function MetricCard({
    icon,
    value,
    label,
    suffix = "",
    delay = 0,
}: MetricCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            whileHover={{ scale: 1.02, y: -1 }}
            className="glass-panel rounded-xl p-4 flex flex-col gap-2 cursor-default group transition-all"
        >
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-secondary">
                <div className="text-muted-foreground opacity-80 group-hover:opacity-100 transition-opacity">
                    {icon}
                </div>
            </div>
            <AnimatedNumber value={value} delay={delay + 0.2} suffix={suffix} />
            <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                {label}
            </span>
        </motion.div>
    );
}
