import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import ollama from 'ollama';
import { encoding_for_model } from '@dqbd/tiktoken';

/* ░ CONFIG ░ */
const MODEL = 'gemma3:27b';
const CTX_LIMIT = 120_000;
const CHUNK_SIZE = 60_000;
const enc = encoding_for_model('gpt-4');

const HEAD = `You are a meeting assistant.
Return:
1. Key decisions
2. Action items (owner • deadline)
3. Open questions
### TRANSCRIPT BELOW
`;

/* ░ helpers ░ */
function* splitTokens(txt: string, size: number, overlap = 0.1) {
  const toks = enc.encode(txt);
  const step = Math.floor(size * (1 - overlap));
  for (let i = 0; i < toks.length; i += step)
    yield enc.decode(toks.slice(i, i + size));
}

async function ask(prompt: string) {
  let out = '';
  for await (const c of await ollama.generate({ model: MODEL, prompt, stream: true })) {
    process.stdout.write(c.response);         // live echo
    out += c.response;
  }
  return out.trim();
}

/* ░ main ░ */
(async () => {
  const [input, output = 'minutes.txt'] = process.argv.slice(2);
  if (!input) {
    console.error('Usage: summarise <transcript.txt> [output.txt]');
    process.exit(1);
  }

  const transcript = await fs.readFile(input, 'utf8');
  const tkCount = enc.encode(transcript).length;

  let minutes: string;

  if (tkCount + enc.encode(HEAD).length < CTX_LIMIT) {
    minutes = await ask(HEAD + transcript);
  } else {
    const parts: string[] = [];
    for (const chunk of splitTokens(transcript, CHUNK_SIZE))
      parts.push(await ask(`Summarise this meeting chunk in ≤200 words:\n###\n${chunk}`));

    minutes = await ask(`Merge the following partial summaries into final minutes (bullets + action items):\n\n${parts.join('\n\n')}`);
  }

  await fs.writeFile(output, minutes + '\n');
  console.log(`\n\n✅  Minutes written to ${output}`);
})();
