import { NextResponse } from 'next/server';

export async function POST(req: Request) {
    try {
        const body = await req.json();

        // Call the Python FastAPI Backend (runs on 8000 locally)
        const backendRes = await fetch('http://127.0.0.1:8000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: body.user_id || "user1",
                message: body.message,
                history: body.history || []
            }),
        });

        if (!backendRes.ok) {
            const errorText = await backendRes.text();
            throw new Error(`Backend Error ${backendRes.status}: ${errorText}`);
        }

        const data = await backendRes.json();
        return NextResponse.json(data);
    } catch (error: any) {
        console.error("Chat API Error:", error);
        return NextResponse.json(
            { error: "Failed to communicate with the reasoning engine.", details: error.message },
            { status: 500 }
        );
    }
}
