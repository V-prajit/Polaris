import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
    process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
    const url = `${BACKEND_URL}/api/polaris/index`;
    const body = await req.json().catch(() => null);

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: body ? { "Content-Type": "application/json" } : {},
            body: body ? JSON.stringify(body) : undefined,
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
