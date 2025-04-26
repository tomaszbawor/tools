import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import ollama from 'ollama';
import { encoding_for_model, TiktokenModel } from '@dqbd/tiktoken';
import readline from 'node:readline/promises';

/* ░ CONFIG ░ */
// Default model, but allow user to override via CLI
const DEFAULT_MODEL = 'gemma3:27b';
const CTX_LIMIT = 120_000;
const CHUNK_SIZE = 60_000;
const DEFAULT_OVERLAP = 0.1;

// Configuration type
interface Config {
  model: string;
  ctxLimit: number;
  chunkSize: number;
  overlap: number;
  systemPrompt: string;
  outputFormat: 'markdown' | 'plain' | 'json';
}

// Default system prompt - more comprehensive with better instructions
const DEFAULT_SYSTEM_PROMPT = `You are a professional meeting assistant.
Analyze the meeting transcript and provide a concise, well-structured summary with the following sections:

1. SUMMARY: Brief overview of the meeting's purpose and main topics (2-3 sentences).
2. KEY DECISIONS: Clear bullet points of decisions made during the meeting.
3. ACTION ITEMS: List tasks assigned to participants with:
   • Owner's name (capitalize)
   • Description of task
   • Deadline when specified
4. OPEN QUESTIONS: Questions that were raised but not resolved.
5. PARTICIPANTS: List of meeting attendees (if identifiable from transcript).

Format everything in clear, professional language. Be specific about deadlines (dates) and owners of tasks.
Focus on extracting factual information, not opinions or side discussions.

### TRANSCRIPT BELOW
`;

/* ░ helpers ░ */
// Get the appropriate tokenizer based on model
function getTokenizer(model: string) {
  // Map Ollama models to tiktoken models
  const modelMap: Record<string, TiktokenModel> = {
    'gemma:2b': 'gpt-3.5-turbo',
    'gemma:7b': 'gpt-3.5-turbo',
    'gemma3:8b': 'gpt-4',
    'gemma3:27b': 'gpt-4',
    'llama3:8b': 'gpt-3.5-turbo',
    'llama3:70b': 'gpt-4',
    'mistral:7b': 'gpt-3.5-turbo',
    'mixtral:8x7b': 'gpt-4',
    'dolphin-mistral': 'gpt-3.5-turbo',
    'phi3:3.8b': 'gpt-3.5-turbo',
    'qwen2:7b': 'gpt-3.5-turbo',
    'qwen2:72b': 'gpt-4'
  };

  // Match prefixes for models not exactly in the map
  let tokenModel: TiktokenModel = 'gpt-4'; // Default to gpt-4 tokenizer
  for (const [prefix, tikModel] of Object.entries(modelMap)) {
    if (model.startsWith(prefix)) {
      tokenModel = tikModel;
      break;
    }
  }

  try {
    return encoding_for_model(tokenModel);
  } catch (error) {
    console.warn(`Warning: Could not load tokenizer for ${tokenModel}, falling back to gpt-4`);
    return encoding_for_model('gpt-4');
  }
}

// Better token chunking with overlap
function* splitTokens(txt: string, enc: any, size: number, overlap = DEFAULT_OVERLAP) {
  const toks = enc.encode(txt);
  const step = Math.floor(size * (1 - overlap));

  if (toks.length <= size) {
    yield txt; // Single chunk case
    return;
  }

  // For multiple chunks, try to split at paragraph boundaries when possible
  for (let i = 0; i < toks.length; i += step) {
    const chunkTokens = toks.slice(i, i + size);
    const chunk = enc.decode(chunkTokens);

    // This chunk is our yield value, but we'll log progress for long transcripts
    if (toks.length > size * 2) {
      const progress = Math.min(100, Math.floor((i + size) / toks.length * 100));
      console.error(`Processing chunk ${Math.floor(i / step) + 1}/${Math.ceil(toks.length / step)} (${progress}%)`);
    }

    yield chunk;
  }
}

// Function to ask Ollama with error handling and retries
async function ask(model: string, prompt: string, systemPrompt?: string, retries = 3): Promise<string> {
  let attempt = 0;
  let out = '';

  while (attempt < retries) {
    try {
      const stream = await ollama.generate({
        model,
        prompt,
        system: systemPrompt,
        stream: true,
        options: {
          temperature: 0.2, // Lower temperature for more factual output
          num_predict: 4096 // Ensure enough tokens for comprehensive summary
        }
      });

      for await (const c of stream) {
        process.stdout.write(c.response);  // Live echo
        out += c.response;
      }

      return out.trim();
    } catch (error) {
      attempt++;
      console.error(`\nError on attempt ${attempt}/${retries}: ${error}`);

      if (attempt < retries) {
        console.error(`Retrying in ${attempt * 2} seconds...`);
        await new Promise(resolve => setTimeout(resolve, attempt * 2000));
      } else {
        console.error(`\nFailed after ${retries} attempts.`);
        throw error;
      }
    }
  }

  return out.trim();
}

// Interactive CLI for specifying options when not provided
async function promptForOptions(): Promise<{ model: string; outputFormat: 'markdown' | 'plain' | 'json' }> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stderr // Use stderr so it doesn't interfere with stdout redirection
  });

  // Show available models first
  try {
    const models = await ollama.list();
    console.error('Available models:');
    models.models.forEach((m: any) => console.error(`- ${m.name}`));
  } catch (error) {
    console.error('Could not fetch available models. Is Ollama running?');
  }

  const model = await rl.question(`Model to use [${DEFAULT_MODEL}]: `) || DEFAULT_MODEL;
  const outputFormat = await rl.question('Output format (markdown, plain, json) [markdown]: ') || 'markdown';

  rl.close();

  return {
    model,
    outputFormat: (outputFormat as 'markdown' | 'plain' | 'json')
  };
}

// Format the output based on specified format
function formatOutput(minutes: string, format: 'markdown' | 'plain' | 'json'): string {
  if (format === 'plain') {
    // Remove markdown formatting
    return minutes
      .replace(/#{1,6} /g, '')
      .replace(/\*\*/g, '')
      .replace(/\*/g, '')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  if (format === 'json') {
    // Basic parsing of the output into JSON structure
    // This is a simple approach and might need refinement for complex outputs
    try {
      const sections: Record<string, any> = {};

      // Extract sections
      const summaryMatch = minutes.match(/#{1,3} SUMMARY\s+([\s\S]*?)(?=#{1,3} KEY|$)/i);
      const decisionsMatch = minutes.match(/#{1,3} KEY DECISIONS\s+([\s\S]*?)(?=#{1,3} ACTION|$)/i);
      const actionsMatch = minutes.match(/#{1,3} ACTION ITEMS\s+([\s\S]*?)(?=#{1,3} OPEN|$)/i);
      const questionsMatch = minutes.match(/#{1,3} OPEN QUESTIONS\s+([\s\S]*?)(?=#{1,3} PARTICIPANTS|$)/i);
      const participantsMatch = minutes.match(/#{1,3} PARTICIPANTS\s+([\s\S]*?)(?=$)/i);

      // Parse out the content for each section
      if (summaryMatch) sections.summary = summaryMatch[1].trim();

      if (decisionsMatch) {
        sections.keyDecisions = decisionsMatch[1]
          .split(/\n[•*-]\s+/)
          .filter(Boolean)
          .map(item => item.trim());
      }

      if (actionsMatch) {
        sections.actionItems = actionsMatch[1]
          .split(/\n[•*-]\s+/)
          .filter(Boolean)
          .map(item => {
            const ownerMatch = item.match(/^([A-Z][A-Za-z\s]+?)[:•]/);
            const deadlineMatch = item.match(/by\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?)|(\d{1,2}\/\d{1,2}(?:\/\d{2,4})?)/i);

            return {
              owner: ownerMatch ? ownerMatch[1].trim() : "Unassigned",
              task: item.replace(/^([A-Z][A-Za-z\s]+?)[:•]/, '').trim(),
              deadline: deadlineMatch ? deadlineMatch[1] || deadlineMatch[2] : null
            };
          });
      }

      if (questionsMatch) {
        sections.openQuestions = questionsMatch[1]
          .split(/\n[•*-]\s+/)
          .filter(Boolean)
          .map(item => item.trim());
      }

      if (participantsMatch) {
        sections.participants = participantsMatch[1]
          .split(/\n[•*-]\s+/)
          .filter(Boolean)
          .map(item => item.trim());
      }

      return JSON.stringify(sections, null, 2);
    } catch (error) {
      console.error(`Error formatting as JSON: ${error}`);
      // Fall back to markdown if JSON parsing fails
      return minutes;
    }
  }

  // Default to markdown
  return minutes;
}

// Check if Ollama is running
async function checkOllamaRunning(): Promise<boolean> {
  try {
    await ollama.list();
    return true;
  } catch (error) {
    return false;
  }
}

// Extract the speaker from a line of transcript
function extractSpeaker(line: string): string | null {
  // Common patterns in meeting transcripts
  const patterns = [
    /^([A-Z][A-Za-z\s.-]+?):\s/,            // Name: text
    /^\[([A-Z][A-Za-z\s.-]+?)\]\s/,         // [Name] text
    /^\(([A-Z][A-Za-z\s.-]+?)\)\s/,         // (Name) text
    /^<([A-Z][A-Za-z\s.-]+?)>\s/            // <Name> text
  ];

  for (const pattern of patterns) {
    const match = line.match(pattern);
    if (match) return match[1];
  }

  return null;
}

// Preprocess transcript to improve summarization quality
async function preprocessTranscript(text: string): Promise<string> {
  const lines = text.split('\n');
  const speakers = new Set<string>();

  // Extract speakers for participants list
  lines.forEach(line => {
    const speaker = extractSpeaker(line);
    if (speaker) speakers.add(speaker);
  });

  // Add metadata about participants at the beginning
  let processedText = text;
  if (speakers.size > 0) {
    const participantsList = Array.from(speakers).join(', ');
    processedText = `PARTICIPANTS: ${participantsList}\n\n${text}`;
  }

  return processedText;
}

/* ░ main ░ */
(async () => {
  // Basic argument parsing with named arguments support
  const args: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (arg.startsWith('--')) {
      const [key, value] = arg.slice(2).split('=');
      args[key] = value || process.argv[++i] || '';
    } else {
      // Positional arguments
      if (!args['input']) args['input'] = arg;
      else if (!args['output']) args['output'] = arg;
    }
  }

  const input = args['input'];
  const output = args['output'] || 'minutes.md'; // Default to .md extension
  const model = args['model'];
  const outputFormatArg = args['format'] as 'markdown' | 'plain' | 'json';
  const ctxLimitArg = args['ctx-limit'] ? parseInt(args['ctx-limit']) : undefined;
  const chunkSizeArg = args['chunk-size'] ? parseInt(args['chunk-size']) : undefined;
  const overlapArg = args['overlap'] ? parseFloat(args['overlap']) : undefined;

  // Show help if requested or if no input file
  if (args['help'] || (!input && process.stdin.isTTY)) {
    console.error(`
Usage: summarise <transcript.txt> [output.md] [options]
       cat transcript.txt | summarise [options]

Options:
  --model=<model>       LLM model to use (default: ${DEFAULT_MODEL})
  --format=<format>     Output format: markdown, plain, or json (default: markdown)
  --ctx-limit=<number>  Context window limit in tokens (default: ${CTX_LIMIT})
  --chunk-size=<number> Size of chunks when splitting (default: ${CHUNK_SIZE})
  --overlap=<number>    Overlap between chunks (0-1) (default: ${DEFAULT_OVERLAP})
  --help                Show this help message

Examples:
  summarise meeting.txt
  summarise meeting.txt summary.md --model=llama3:70b
  cat meeting.txt | summarise --format=json > summary.json
    `);
    process.exit(args['help'] ? 0 : 1);
  }

  // Check if Ollama is running
  if (!await checkOllamaRunning()) {
    console.error('Error: Ollama is not running. Please start Ollama first.');
    process.exit(1);
  }

  // Interactive mode if no input arguments specified
  const options = process.stdin.isTTY && !input ? await promptForOptions() : {
    model: DEFAULT_MODEL,
    outputFormat: 'markdown' as const
  };

  // Set final config values, prioritizing command line > interactive > defaults
  const finalConfig: Config = {
    model: model || options.model || DEFAULT_MODEL,
    ctxLimit: ctxLimitArg || CTX_LIMIT,
    chunkSize: chunkSizeArg || CHUNK_SIZE,
    overlap: overlapArg !== undefined ? overlapArg : DEFAULT_OVERLAP,
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    outputFormat: outputFormatArg || options.outputFormat || 'markdown'
  };

  console.error(`Using model: ${finalConfig.model}`);

  // Get the appropriate tokenizer
  const enc = getTokenizer(finalConfig.model);

  // Handle piped input or read from file
  let transcript: string;
  if (!input && !process.stdin.isTTY) {
    // Read from stdin (piped input)
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
    transcript = Buffer.concat(chunks).toString('utf8');
  } else {
    // Read from file
    transcript = await fs.readFile(input, 'utf8');
  }

  // Preprocess transcript
  transcript = await preprocessTranscript(transcript);

  // Calculate token count
  const tkCount = enc.encode(transcript).length;
  const systemPromptTokens = enc.encode(finalConfig.systemPrompt).length;

  console.error(`Transcript: ${tkCount} tokens`);

  let minutes: string;

  if (tkCount + systemPromptTokens < finalConfig.ctxLimit) {
    console.error('Processing entire transcript at once...');
    minutes = await ask(finalConfig.model, transcript, finalConfig.systemPrompt);
  } else {
    console.error(`Transcript exceeds context window (${finalConfig.ctxLimit}), processing in chunks...`);

    const parts: string[] = [];
    let chunkNum = 1;
    for (const chunk of splitTokens(transcript, enc, finalConfig.chunkSize, finalConfig.overlap)) {
      console.error(`\nProcessing chunk ${chunkNum++}...`);
      parts.push(await ask(
        finalConfig.model,
        `Summarize this meeting chunk extracting key points, decisions, and action items:\n\n${chunk}`,
        'Extract only factual information from this meeting transcript chunk. Focus on decisions made, action items assigned, and questions raised.'
      ));
    }

    console.error('\nMerging summaries...');
    minutes = await ask(
      finalConfig.model,
      `Merge these partial meeting summaries into a final, well-structured meeting minutes document:\n\n${parts.join('\n\n===NEXT CHUNK===\n\n')}`,
      `You are a professional meeting assistant. Create a comprehensive, well-structured meeting summary from these partial summaries. Include:
1. SUMMARY: Brief overview of meeting purpose (2-3 sentences)
2. KEY DECISIONS: Clear bullet points of decisions made
3. ACTION ITEMS: List with owner names (CAPITALIZED), task description, and deadlines
4. OPEN QUESTIONS: Questions raised but not resolved
5. PARTICIPANTS: List of attendees (if mentioned)

Format in clear, professional markdown. Eliminate any redundancy from the partial summaries.`
    );
  }

  // Format output according to specified format
  const formattedOutput = formatOutput(minutes, finalConfig.outputFormat as 'markdown' | 'plain' | 'json');

  // Write to file or stdout
  if (output !== '-') {
    await fs.writeFile(output, formattedOutput + '\n');
    console.error(`\n\n✅ Minutes written to ${output}`);
  } else {
    // If output is '-', write to stdout (useful for piping)
    console.log(formattedOutput);
  }
})().catch(error => {
  console.error(`\n❌ Error: ${error.message}`);
  process.exit(1);
});
