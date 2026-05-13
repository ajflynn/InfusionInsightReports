import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "25mb" }));

// Serve frontend
app.use(express.static(__dirname));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "report-generator.html"));
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    message: "Backend running",
  });
});

// Main endpoint
app.post("/api/generate-report", async (req, res) => {
  try {
    const { provider, model, system, prompt } = req.body;

    if (!provider || !model || !prompt) {
      return res.status(400).json({
        error: "Missing provider/model/prompt",
      });
    }

    const text = await callLLM({
      provider,
      model,
      system,
      prompt,
    });

    res.json({ text });
  } catch (err) {
    console.error(err);

    res.status(500).json({
      error: err.message || "Server error",
    });
  }
});

async function callLLM({ provider, model, system, prompt }) {
  switch (provider) {
    case "anthropic":
      return callAnthropic({ model, system, prompt });

    case "openai":
      return callOpenAI({ model, system, prompt });

    case "google":
      return callGoogle({ model, system, prompt });

    case "ollama":
      return callOllama({ model, system, prompt });

    default:
      throw new Error(`Unsupported provider: ${provider}`);
  }
}

async function callAnthropic({ model, system, prompt }) {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error("Missing ANTHROPIC_API_KEY");
  }

  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const response = await anthropic.messages.create({
    model,
    max_tokens: 4000,
    system: system || "",
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  });

  return response.content
    .map((x) => x.text || "")
    .join("")
    .trim();
}

async function callOpenAI({ model, system, prompt }) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("Missing OPENAI_API_KEY");
  }

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const response = await openai.chat.completions.create({
    model,
    temperature: 0.2,
    messages: [
      {
        role: "system",
        content: system || "You are a helpful assistant.",
      },
      {
        role: "user",
        content: prompt,
      },
    ],
  });

  return response.choices?.[0]?.message?.content?.trim() || "";
}

async function callGoogle({ model, system, prompt }) {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("Missing GOOGLE_API_KEY");
  }

  const genAI = new GoogleGenerativeAI(
    process.env.GOOGLE_API_KEY
  );

  const geminiModel = genAI.getGenerativeModel({
    model,
    systemInstruction: system || undefined,
  });

  const result = await geminiModel.generateContent(prompt);

  return result.response.text().trim();
}

async function callOllama({ model, system, prompt }) {
  const baseUrl =
    process.env.OLLAMA_BASE_URL ||
    "http://localhost:11434";

  const response = await fetch(
    `${baseUrl}/api/generate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        prompt: `${system || ""}\n\n${prompt}`,
        stream: false,
      }),
    }
  );

  if (!response.ok) {
    const txt = await response.text();
    throw new Error(`Ollama error: ${txt}`);
  }

  const data = await response.json();

  return data.response?.trim() || "";
}

app.listen(PORT, () => {
  console.log(
    `Infusion Insight Reports app running at http://localhost:${PORT}`
  );
});
