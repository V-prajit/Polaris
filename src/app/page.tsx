"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Globe } from "@/components/ui/globe";
import { Progress, ProgressIndicator } from "@/components/animate-ui/primitives/radix/progress";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function Home() {
  const router = useRouter();
  const [location, setLocation] = useState("");
  const [selectedLat, setSelectedLat] = useState("");
  const [selectedLng, setSelectedLng] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");

  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isSearchingSuggestions, setIsSearchingSuggestions] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (!location.trim() || location.length < 2) {
      setSuggestions([]);
      return;
    }

    if (selectedLat && selectedLng) return;

    const ATLANTA_CENTER = { lat: 33.749, lon: -84.388 };
    const PHOTON_BBOX = "-84.7,33.5,-84.1,34.0";

    const timeoutId = setTimeout(async () => {
      setIsSearchingSuggestions(true);
      try {
        const params = new URLSearchParams({
          q: location,
          limit: "10",
          lat: String(ATLANTA_CENTER.lat),
          lon: String(ATLANTA_CENTER.lon),
          bbox: PHOTON_BBOX,
          lang: "en",
        });

        const res = await fetch(`https://photon.komoot.io/api/?${params.toString()}`);
        const data = await res.json();

        const uniqueSuggestions: any[] = [];
        const seenNames = new Set<string>();

        for (const feature of (data?.features || [])) {
          const props = feature.properties;
          const coords = feature.geometry.coordinates; // [lon, lat]

          const name = props.name || props.street || "";
          const city = props.city || props.town || props.village || "";
          const displayName = [name, city, "GA"].filter(Boolean).join(", ");

          if (!name || seenNames.has(displayName)) continue;
          seenNames.add(displayName);

          uniqueSuggestions.push({
            display_name: displayName,
            name,
            lat: String(coords[1]),
            lon: String(coords[0]),
          });
        }

        setSuggestions(uniqueSuggestions.slice(0, 3));
      } catch (err) {
        console.error("Autocomplete error:", err);
      } finally {
        setIsSearchingSuggestions(false);
      }
    }, 250);

    return () => clearTimeout(timeoutId);
  }, [location, selectedLat, selectedLng]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!location.trim()) return;

    setIsSubmitting(true);
    setProgress(0);
    setErrorMsg("");

    let finalLat = selectedLat;
    let finalLng = selectedLng;

    if (!finalLat || !finalLng) {
      const query = location.toLowerCase().includes("atlanta")
        ? location
        : `${location}, Atlanta, GA`;

      try {
        const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`);
        const data = await res.json();
        if (data && data.length > 0) {
          finalLat = data[0].lat;
          finalLng = data[0].lon;
        } else {
          setErrorMsg("Location not found. Try another place in Atlanta.");
          setIsSubmitting(false);
          return;
        }
      } catch (err) {
        console.error(err);
        setErrorMsg("Error finding location.");
        setIsSubmitting(false);
        return;
      }
    }

    // Adobe-style smooth easing progress simulation
    const duration = 2500;
    const interval = 20;
    let elapsed = 0;

    const timer = setInterval(() => {
      elapsed += interval;
      // Easing function (easeOutExpo)
      const p = Math.min(elapsed / duration, 1);
      const easedP = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);

      setProgress(Math.min(easedP * 100, 100));

      if (elapsed >= duration) {
        clearInterval(timer);
        setTimeout(() => {
          router.push(`/map?lat=${finalLat}&lng=${finalLng}`);
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
          className="relative z-10 mt-16 mb-0 text-center text-7xl font-bold tracking-tight md:text-8xl select-none"
          style={{
            background: "linear-gradient(180deg, hsl(var(--foreground)) 0%, transparent 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}
        >
          Polaris
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

      {/* Search — centered under the title */}
      <AnimatePresence mode="wait">
        {!isSubmitting ? (
          <motion.form
            key="form"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onSubmit={handleSubmit}
            className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 flex w-full max-w-md flex-col items-center gap-2 mt-[-5vh]"
          >
            <div className="flex w-full gap-2 relative" ref={wrapperRef}>
              <Input
                type="text"
                placeholder="Enter a location in Atlanta (e.g. Piedmont Park)"
                value={location}
                onChange={(e) => {
                  setLocation(e.target.value);
                  setSelectedLat("");
                  setSelectedLng("");
                  setShowSuggestions(true);
                }}
                onFocus={() => setShowSuggestions(true)}
                className="flex-1 h-9 text-sm bg-background/80 backdrop-blur-sm"
              />
              <AnimatePresence>
                {showSuggestions && (suggestions.length > 0 || isSearchingSuggestions) && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute top-[100%] mt-1 left-0 w-full bg-background border border-border/80 rounded-lg shadow-xl overflow-hidden z-50 text-sm max-h-60 overflow-y-auto"
                  >
                    {isSearchingSuggestions && suggestions.length === 0 && (
                      <div className="px-3 py-2 text-xs text-muted-foreground flex items-center justify-center h-10">
                        Searching...
                      </div>
                    )}
                    {suggestions.map((s, i) => {
                      const mainText = s.name || s.display_name.split(",")[0];
                      const subText = s.display_name;
                      return (
                        <button
                          key={i}
                          type="button"
                          className="w-full text-left px-3 py-2 hover:bg-secondary/80 focus:bg-secondary/80 outline-none transition-colors border-b border-border/40 last:border-0"
                          onClick={() => {
                            setLocation(mainText);
                            setSelectedLat(s.lat);
                            setSelectedLng(s.lon);
                            setShowSuggestions(false);
                          }}
                        >
                          <p className="font-medium truncate text-foreground text-xs leading-none mb-1">{mainText}</p>
                          <p className="text-[10px] text-muted-foreground line-clamp-1 leading-tight">{subText}</p>
                        </button>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
              <Button type="submit" className="h-9 px-6 text-sm flex-shrink-0" disabled={!location.trim()}>
                Search
              </Button>
            </div>
            {errorMsg && <p className="text-red-500 text-xs w-full text-right">{errorMsg}</p>}
          </motion.form>
        ) : (
          <motion.div
            key="progress"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-10 right-10 z-30 w-60 space-y-4"
          >
            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">
                Getting you the most accurate estimate...
              </p>
              <Progress value={progress} className="w-full h-2 border overflow-hidden">
                <ProgressIndicator className="size-full flex-1 bg-primary" />
              </Progress>
            </div>
            <p className="text-xs text-muted-foreground text-center animate-pulse">
              Powered by GrowthFactor
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
