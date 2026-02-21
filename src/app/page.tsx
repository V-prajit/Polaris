"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Globe } from "@/components/ui/globe";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function Home() {
  const router = useRouter();
  const [lat, setLat] = useState("33.8025746");
  const [lng, setLng] = useState("-84.4106416");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!lat || !lng) return;

    setIsSubmitting(true);
    setProgress(0);

    // Adobe-style smooth easing progress simulation
    const duration = 2500;
    const interval = 20;
    let elapsed = 0;

    const timer = setInterval(() => {
      elapsed += interval;
      // Easing function (easeOutExpo)
      const p = elapsed / duration;
      const easedP = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);

      setProgress(Math.min(easedP * 100, 100));

      if (elapsed >= duration) {
        clearInterval(timer);
        setTimeout(() => {
          router.push(`/map?lat=${lat}&lng=${lng}`);
        }, 300);
      }
    }, interval);
  };

  return (
    <div className="relative min-h-screen bg-background overflow-hidden">
      {/* Globe with title and glow */}
      <div className="relative w-full flex flex-col items-center" style={{ height: "85vh" }}>
        {/* Title — behind the glow */}
        <h1
          className="relative z-10 mt-16 mb-0 text-center text-7xl font-bold tracking-tight md:text-8xl"
          style={{
            background: "linear-gradient(180deg, hsl(var(--foreground)) 0%, transparent 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}
        >
          parkIQ
        </h1>

        {/* Globe */}
        <div className="relative flex-1 w-full flex items-center justify-center" style={{ marginTop: "-30px" }}>
          <Globe className="max-w-none" />
        </div>

        {/* Cloudy/blurry glow radiating upward from globe — on top of text */}
        <div
          className="pointer-events-none absolute inset-0 z-20"
          style={{
            background: "radial-gradient(ellipse 80% 40% at 50% 65%, hsl(var(--background) / 0.95) 0%, hsl(var(--background) / 0.6) 20%, transparent 55%)",
            filter: "blur(25px)",
          }}
        />
      </div>

      {/* Search — fixed bottom right of the page */}
      <AnimatePresence mode="wait">
        {!isSubmitting ? (
          <motion.form
            key="form"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onSubmit={handleSubmit}
            className="absolute bottom-10 right-10 z-30 flex w-full max-w-xs flex-col items-end gap-2"
          >
            <div className="flex w-full gap-2">
              <Input
                type="number"
                step="any"
                placeholder="Latitude"
                value={lat}
                onChange={(e) => setLat(e.target.value)}
                className="flex-1 h-9 text-sm bg-background/80 backdrop-blur-sm"
              />
              <Input
                type="number"
                step="any"
                placeholder="Longitude"
                value={lng}
                onChange={(e) => setLng(e.target.value)}
                className="flex-1 h-9 text-sm bg-background/80 backdrop-blur-sm"
              />
            </div>
            <Button type="submit" className="w-full h-9 text-sm" disabled={!lat || !lng}>
              Search
            </Button>
          </motion.form>
        ) : (
          <motion.div
            key="progress"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-10 right-10 z-30 w-60 space-y-4"
          >
            <div className="space-y-2">
              <div className="flex justify-between text-sm font-medium text-muted-foreground">
                <span>Processing imagery...</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
            <p className="text-xs text-muted-foreground text-center animate-pulse">
              Running SegFormer-B4 segmentation
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
