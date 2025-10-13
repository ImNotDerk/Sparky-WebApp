import { NextRequest, NextResponse } from "next/server";

type GenerateRequestBody = {
  prompt: string;
};

export async function POST(req: NextRequest) {
  const { prompt } = await req.json();

  const response = await fetch("http://127.0.0.1:8000/send_message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const data = await response.json();
  return NextResponse.json(data);
}
