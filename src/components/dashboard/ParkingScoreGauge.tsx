"use client";

import { motion } from "framer-motion";

interface ParkingScoreGaugeProps {
    score: number; // 0-100
    label?: string;
    delay?: number;
}

export function ParkingScoreGauge({
    score,
    label = "ParkIQ Score",
    delay = 0,
}: ParkingScoreGaugeProps) {
    const radius = 58;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (score / 100) * circumference;

    const getScoreColor = (s: number) => {
        if (s >= 75) return "hsl(142, 71%, 45%)";
        if (s >= 50) return "hsl(187, 73%, 46%)";
        if (s >= 25) return "hsl(38, 92%, 50%)";
        return "hsl(0, 84%, 60%)";
    };

    const getGrade = (s: number) => {
        if (s >= 85) return "Excellent";
        if (s >= 70) return "Good";
        if (s >= 50) return "Fair";
        return "Limited";
    };

    const color = getScoreColor(score);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col items-center gap-3"
        >
            <div className="relative w-36 h-36">
                <svg
                    className="w-full h-full -rotate-90"
                    viewBox="0 0 140 140"
                >
                    {/* Background track */}
                    <circle
                        cx="70"
                        cy="70"
                        r={radius}
                        fill="none"
                        stroke="hsla(215, 25%, 20%, 0.5)"
                        strokeWidth="10"
                        strokeLinecap="round"
                    />
                    {/* Score arc */}
                    <motion.circle
                        cx="70"
                        cy="70"
                        r={radius}
                        fill="none"
                        stroke={color}
                        strokeWidth="10"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset }}
                        transition={{
                            delay: delay + 0.3,
                            duration: 1.4,
                            ease: [0.16, 1, 0.3, 1],
                        }}
                        style={{
                            filter: `drop-shadow(0 0 8px ${color})`,
                        }}
                    />
                </svg>
                {/* Center content */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <motion.span
                        className="text-3xl font-bold tracking-tight"
                        style={{ color }}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: delay + 0.8, duration: 0.5 }}
                    >
                        {score}
                    </motion.span>
                    <motion.span
                        className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: delay + 1, duration: 0.5 }}
                    >
                        {getGrade(score)}
                    </motion.span>
                </div>
            </div>
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {label}
            </span>
        </motion.div>
    );
}
