import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
    process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(req: NextRequest) {
    const { searchParams } = req.nextUrl;
    const lat = searchParams.get("lat");
    const lon = searchParams.get("lon");
    const radius = searchParams.get("radius") || "300";

    if (!lat || !lon) {
        return NextResponse.json(
            { error: "lat and lon are required" },
            { status: 400 },
        );
    }

    const url = `${BACKEND_URL}/api/estimate?lat=${lat}&lon=${lon}&radius=${radius}`;

    try {
        const res = await fetch(url, {
            signal: AbortSignal.timeout(120_000),
        });

        if (!res.ok) {
            const text = await res.text().catch(() => "");
            return NextResponse.json(
                { error: text || res.statusText },
                { status: res.status },
            );
        }

        const data = await res.json();
        return NextResponse.json(data);
    } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        return NextResponse.json(
            { error: `Backend unreachable: ${msg}` },
            { status: 502 },
        );
    }
}
