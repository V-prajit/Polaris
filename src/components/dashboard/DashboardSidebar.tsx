"use client";

import { HomeLine, BookmarkCheck, Settings01, ChevronSelectorVertical } from "@untitledui/icons";
import { PanelLeft } from "lucide-react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";

const NAV_TOP = [
  { label: "Home", href: "/", icon: HomeLine },
  { label: "Saved Searches", href: "/saved", icon: BookmarkCheck },
];

const NAV_BOTTOM = [
  { label: "Settings", href: "/settings", icon: Settings01 },
];

// ─── Standalone toggle button (IDE-style, placed externally) ─────────────────
export function SidebarToggleButton({ onClick }: { onClick: () => void }) {
  return (
    <div className="relative group">
      <button
        onClick={onClick}
        className="flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-background shadow-sm hover:bg-secondary transition-colors"
        aria-label="Toggle sidebar"
      >
        <PanelLeft className="h-[17px] w-[17px] text-muted-foreground" />
      </button>
      {/* Tooltip */}
      <div className="pointer-events-none absolute left-1/2 top-[calc(100%+6px)] -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50">
        <div className="flex items-center gap-2 whitespace-nowrap rounded-lg border border-border bg-background px-3 py-1.5 shadow-md">
          <span className="text-[12px] font-medium text-foreground">Toggle sidebar</span>
          <kbd className="flex items-center justify-center rounded-md bg-secondary px-1.5 py-0.5 text-[11px] font-mono text-muted-foreground border border-border">
            ⌘B
          </kbd>
        </div>
      </div>
    </div>
  );
}

// ─── Sidebar panel ────────────────────────────────────────────────────────────
interface DashboardSidebarProps {
  expanded: boolean;
}

export function DashboardSidebar({ expanded }: DashboardSidebarProps) {
  const pathname = usePathname();

  const NavItem = ({ href, icon: Icon, label }: { href: string; icon: any; label: string }) => {
    const active = pathname === href;
    return (
      <Link
        href={href}
        title={!expanded ? label : undefined}
        className={`flex items-center gap-3 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors
          ${active
            ? "bg-secondary text-foreground"
            : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
          }
          ${!expanded ? "justify-center px-0" : ""}
        `}
      >
        <Icon className="h-[17px] w-[17px] flex-shrink-0" strokeWidth={1.8} />
        {expanded && <span className="truncate">{label}</span>}
      </Link>
    );
  };

  return (
    <div
      className="absolute left-3 top-3 bottom-3 z-30 flex flex-col bg-background border border-border rounded-2xl shadow-sm transition-[width] duration-200 ease-in-out overflow-hidden"
      style={{ width: expanded ? 216 : 50 }}
    >
      {/* ── Logo ── */}
      <div className={`flex h-[58px] items-center flex-shrink-0 gap-2.5 ${expanded ? "px-4" : "justify-center"}`}>
        <Image
          src="/globe-logo.png"
          alt="Polaris"
          width={28}
          height={28}
          className="rounded-lg flex-shrink-0 object-cover"
        />
        {expanded && (
          <span className="text-[14px] font-bold tracking-tight text-foreground">Polaris</span>
        )}
      </div>

      {/* ── Divider ── */}
      <div className="h-px bg-border mx-3 flex-shrink-0" />

      {/* ── Main nav ── */}
      <nav className={`flex-1 py-3 space-y-0.5 overflow-hidden ${expanded ? "px-2.5" : "px-1.5"}`}>
        {NAV_TOP.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}
      </nav>

      {/* ── Divider ── */}
      <div className="h-px bg-border mx-3 flex-shrink-0" />

      {/* ── Bottom nav ── */}
      <div className={`py-3 space-y-0.5 ${expanded ? "px-2.5" : "px-1.5"}`}>
        {NAV_BOTTOM.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}
      </div>

      {/* ── Divider ── */}
      <div className="h-px bg-border mx-3 flex-shrink-0" />

      {/* ── User card ── */}
      <div className={`flex-shrink-0 ${expanded ? "p-3" : "py-3 flex justify-center"}`}>
        <div className={`flex items-center gap-2.5 rounded-xl border border-border transition-colors hover:bg-secondary/60 cursor-pointer ${expanded ? "p-2.5 pr-2" : "p-1.5"}`}>
          {/* Avatar */}
          <div className="h-8 w-8 rounded-full bg-foreground flex items-center justify-center flex-shrink-0 ring-2 ring-border">
            <span className="text-background text-[12px] font-bold">P</span>
          </div>
          {expanded && (
            <>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-semibold text-foreground truncate leading-tight">Polaris</p>
                <p className="text-[11px] text-muted-foreground truncate leading-tight">polaris@growthfactor.ai</p>
              </div>
              <ChevronSelectorVertical className="h-4 w-4 text-muted-foreground flex-shrink-0" strokeWidth={1.8} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
