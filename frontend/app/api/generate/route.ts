import { NextRequest, NextResponse } from "next/server";

type GenerateRequestBody = {
  prompt: string;
};

export async function POST(req: NextRequest) { 
  const { prompt } = await req.json();

  const reponse = await fetch("http://127.0.0.1:8000/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });

  const data = await reponse.json();
  // console.log("Backend Response:", data);
  return NextResponse.json(data);
}
