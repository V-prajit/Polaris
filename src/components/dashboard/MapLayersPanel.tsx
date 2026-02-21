"use client";

import { motion } from "framer-motion";
import { Eye, EyeOff, Layers, MapPin, Route, Circle, Satellite, MapIcon } from "lucide-react";
import { useState } from "react";

interface LayerToggle {
    id: string;
    label: string;
    icon: React.ReactNode;
    defaultOn: boolean;
    count?: number;
}

interface MapLayersPanelProps {
    onToggle: (id: string, visible: boolean) => void;
    delay?: number;
}

const LAYER_GROUPS: { title: string; layers: LayerToggle[] }[] = [
    {
        title: "ANALYSIS",
        layers: [
            { id: "parking", label: "Parking Zones", icon: <MapPin className="w-3.5 h-3.5" />, defaultOn: true, count: 2 },
            { id: "roads", label: "Road Segments", icon: <Route className="w-3.5 h-3.5" />, defaultOn: true, count: 7 },
            { id: "radius", label: "Search Radius", icon: <Circle className="w-3.5 h-3.5" />, defaultOn: true },
        ],
    },
    {
        title: "MAP STYLE",
        layers: [
            { id: "satellite", label: "Satellite", icon: <Satellite className="w-3.5 h-3.5" />, defaultOn: true },
            { id: "labels", label: "Street Labels", icon: <MapIcon className="w-3.5 h-3.5" />, defaultOn: true },
        ],
    },
];

export function MapLayersPanel({ onToggle, delay = 0 }: MapLayersPanelProps) {
    const [isOpen, setIsOpen] = useState(true);
    const [layerState, setLayerState] = useState<Record<string, boolean>>(() => {
        const state: Record<string, boolean> = {};
        LAYER_GROUPS.forEach((g) =>
            g.layers.forEach((l) => {
                state[l.id] = l.defaultOn;
            })
        );
        return state;
    });

    const toggle = (id: string) => {
        const next = !layerState[id];
        setLayerState((s) => ({ ...s, [id]: next }));
        onToggle(id, next);
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="glass-panel-strong rounded-2xl overflow-hidden w-56"
        >
            {/* Header */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-secondary/30 transition-colors"
            >
                <span className="flex items-center gap-2 text-sm font-semibold">
                    <Layers className="w-4 h-4 text-primary" />
                    Layers
                </span>
                <motion.span
                    animate={{ rotate: isOpen ? 0 : -90 }}
                    transition={{ duration: 0.2 }}
                    className="text-muted-foreground text-xs"
                >
                    ▼
                </motion.span>
            </button>

            {/* Body */}
            {isOpen && (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    transition={{ duration: 0.3 }}
                    className="px-3 pb-3 space-y-3"
                >
                    {LAYER_GROUPS.map((group) => (
                        <div key={group.title}>
                            <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-muted-foreground/60 px-1 mb-1.5">
                                {group.title}
                            </div>
                            <div className="space-y-0.5">
                                {group.layers.map((layer) => {
                                    const on = layerState[layer.id];
                                    return (
                                        <button
                                            key={layer.id}
                                            onClick={() => toggle(layer.id)}
                                            className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs transition-all ${on
                                                    ? "bg-primary/10 text-foreground"
                                                    : "text-muted-foreground hover:bg-secondary/30"
                                                }`}
                                        >
                                            <span
                                                className="transition-colors"
                                                style={{ color: on ? "hsl(187, 73%, 46%)" : undefined }}
                                            >
                                                {layer.icon}
                                            </span>
                                            <span className="flex-1 text-left font-medium">
                                                {layer.label}
                                            </span>
                                            {layer.count != null && (
                                                <span className="text-[10px] tabular-nums text-muted-foreground/50">
                                                    {layer.count}
                                                </span>
                                            )}
                                            {on ? (
                                                <Eye className="w-3 h-3 text-primary/60" />
                                            ) : (
                                                <EyeOff className="w-3 h-3 text-muted-foreground/40" />
                                            )}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </motion.div>
            )}
        </motion.div>
    );
}
