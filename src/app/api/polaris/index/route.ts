import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
    process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(_req: NextRequest) {
    const url = `${BACKEND_URL}/api/polaris/index`;

    try {
        const res = await fetch(url, {
            method: "POST",
            signal: AbortSignal.timeout(300_000),
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
