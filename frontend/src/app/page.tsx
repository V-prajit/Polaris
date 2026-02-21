"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Globe } from "@/components/magicui/globe";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/animate-ui/components/buttons/button";
import { Progress } from "@/components/ui/progress";
import { ArrowRight } from "lucide-react";

export default function Home() {
  const router = useRouter();
  const [lat, setLat] = useState("33.8025746");
  const [lng, setLng] = useState("-84.4106416");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);

  const MAX_CHARS = 12;

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
        }, 300); // Slight delay before redirecting
      }
    }, interval);
  };

  return (
    <main className="relative min-h-screen bg-background overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(12,18,28,0.18),transparent_55%)]" />

      {/* Bigger globe */}
      <div className="absolute inset-0 flex items-center justify-center opacity-70 pointer-events-none">
        <Globe className="top-10" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_210%,rgba(0,0,0,0.28),rgba(255,255,255,0))] dark:bg-[radial-gradient(circle_at_50%_210%,rgba(255,255,255,0.14),rgba(0,0,0,0))]" />
      </div>

      {/* Title above globe */}
      <div className="relative z-20 flex flex-col items-center pt-10 sm:pt-14">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="bg-gradient-to-b from-black via-zinc-500 to-zinc-300/20 bg-clip-text text-center text-6xl sm:text-8xl leading-none font-bold tracking-tighter text-transparent dark:from-white dark:via-zinc-300 dark:to-zinc-900/10"
        >
          ParkIQ
        </motion.h1>
        <div className="mt-2 h-6 w-56 rounded-full bg-zinc-400/45 blur-2xl dark:bg-white/25" />
      </div>

      {/* Inputs bottom-right, no outer card box */}
      <div className="absolute bottom-6 right-6 left-6 sm:left-auto sm:w-[360px] z-20">
        <AnimatePresence mode="wait">
          {!isSubmitting ? (
            <motion.form
              key="form"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              onSubmit={handleSubmit}
              className="space-y-4"
            >
              <div className="relative">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1 block">
                  Latitude
                </label>
                <Input
                  type="text"
                  placeholder="e.g. 33.8025"
                  value={lat}
                  maxLength={MAX_CHARS}
                  onChange={(e) => setLat(e.target.value.replace(/[^0-9.-]/g, ""))}
                  className="pr-12 text-lg h-12 bg-background/45 border-white/30"
                />
                <span className="absolute right-3 bottom-3 text-xs text-muted-foreground/60">
                  {MAX_CHARS - lat.length}
                </span>
              </div>

              <div className="relative">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1 block">
                  Longitude
                </label>
                <Input
                  type="text"
                  placeholder="e.g. -84.4106"
                  value={lng}
                  maxLength={MAX_CHARS}
                  onChange={(e) => setLng(e.target.value.replace(/[^0-9.-]/g, ""))}
                  className="pr-12 text-lg h-12 bg-background/45 border-white/30"
                />
                <span className="absolute right-3 bottom-3 text-xs text-muted-foreground/60">
                  {MAX_CHARS - lng.length}
                </span>
              </div>

              <Button
                type="submit"
                variant="default"
                size="lg"
                disabled={!lat || !lng}
                className="w-full"
              >
                Analyze
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </motion.form>
          ) : (
            <motion.div
              key="progress"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
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
    </main>
  );
}
