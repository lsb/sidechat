// Prompt-optimization harness for the list/prose classifier.
//
// We define:
//   - 50 "list" prompts (answer best rendered as a bulleted/numbered list)
//   - 50 "prose" prompts (answer best rendered as narrative/explanation/etc.)
// First 10 of each are held out as a validation set; the remaining 40+40 are
// the dev set we iterate variants on.
//
// A `variant` is `{ name, system, prefill, branches, parse }`. We apply the
// chat template with the variant's system + the user prompt, append `prefill`
// to the resulting string, and constrain generation to exactly one of
// `branches` (literal text) via unionGrammars of compileLiteral. The grammar
// is tiny so scanning ~49k tokens is fast; the dominant cost is the model
// forward pass (~1 s prefill + a handful of generated tokens).

import { LogitsProcessorList, TextStreamer } from '@huggingface/transformers';
import { compileLiteral, unionGrammars } from './grammar.js';
import { GrammarLogitsProcessor } from './logits.js';

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export const LIST_PROMPTS = [
  // --- validation (first 10) ---
  'list 10 ways to improve morale at work',
  'give me five reasons to learn Rust',
  'what are the main benefits of meditation?',
  'suggest some names for my new puppy',
  'name three famous jazz musicians',
  'list the ingredients for guacamole',
  'what are the steps to change a tire?',
  'give me ideas for weekend activities with kids',
  'tips for packing light when traveling',
  'what are some common Italian desserts?',
  // --- dev (next 40) ---
  'list popular video game consoles from the 1990s',
  'suggest questions to ask at a job interview',
  'what are the symptoms of dehydration?',
  'name ten countries in Africa',
  'list some movies directed by Christopher Nolan',
  'give me seven examples of onomatopoeia',
  'what tools do I need to build a raised garden bed?',
  'suggest some icebreaker activities for a team meeting',
  'ways to reduce food waste at home',
  'list the planets in order from the sun',
  'what are the main differences between Python 2 and Python 3?',
  'give me 5 good podcast recommendations about history',
  'name three types of dance',
  'top tourist attractions in Kyoto',
  'list common symptoms of the flu',
  'what are some healthy snack ideas for kids?',
  'suggest some books similar to The Hobbit',
  'name five spices commonly used in Indian cooking',
  'list programming languages that compile to WebAssembly',
  'give me a list of yoga poses for beginners',
  'what are some good stretches before running?',
  'name the colors of the rainbow',
  'list the months of the year in French',
  'what are common causes of burnout?',
  'suggest some romantic date ideas in New York',
  'give me a bullet list of home safety tips',
  'list the bones in the human hand',
  'ways to learn a new language quickly',
  'name five mammals native to Australia',
  'what are some highlights of the French Revolution?',
  'list common pitfalls of distributed systems',
  'top 10 songs from the 1980s',
  'suggest some hobbies for introverts',
  'name the original members of The Beatles',
  'what are the primary colors?',
  'list reasons to adopt a cat',
  'give me 6 tips for better sleep hygiene',
  'name the Great Lakes',
  'list programming concepts every developer should know',
  'suggest some vegan dinner recipes',
];

export const PROSE_PROMPTS = [
  // --- validation (first 10) ---
  'tell me a short story about a lighthouse keeper',
  'write a haiku about autumn',
  'explain how a solar panel works in a paragraph',
  'summarize the plot of Pride and Prejudice',
  'what does the word "quixotic" mean?',
  'translate "good morning" to Japanese',
  'write a professional email declining a meeting',
  'describe the taste of a ripe mango',
  'compose a poem about loneliness',
  'what is the capital of Australia?',
  // --- dev (next 40) ---
  'tell me about the invention of the printing press',
  'write a cover letter for a software engineering role',
  'explain the theory of relativity to a 10-year-old',
  'who was Marie Curie?',
  'describe a sunset over the ocean',
  'what is photosynthesis?',
  'write a bedtime story for a 4-year-old',
  'explain how blockchain works',
  'tell me about the history of tea in China',
  'describe the plot of Inception',
  'write a haiku about the sea',
  'what is the meaning of life according to Camus?',
  'tell me a joke about programming',
  'explain why the sky is blue',
  'describe what it feels like to run a marathon',
  'write a love letter in the style of Shakespeare',
  'what year did the Berlin Wall fall?',
  'tell me about the architecture of the Sagrada Familia',
  'write a persuasive essay on renewable energy',
  'describe the personality of a golden retriever',
  'who was the first person on the moon?',
  'tell me about quantum entanglement briefly',
  'write a one-paragraph synopsis of The Great Gatsby',
  'what is the etymology of the word "sandwich"?',
  'explain why we dream',
  'tell me a myth about the origin of fire',
  'describe the feeling of nostalgia',
  'write a toast for a wedding',
  'what does "serendipity" mean?',
  'tell me about your favorite season',
  'explain the difference between empathy and sympathy',
  'who wrote Hamlet?',
  'write a limerick about cats',
  'tell me a ghost story',
  'describe Mount Fuji in winter',
  'what happened in the Cuban Missile Crisis?',
  'explain how a car engine works',
  'tell me a folk tale from Ireland',
  'write an essay on the importance of libraries',
  'describe a perfect day',
];

export const VALIDATION_LIST = LIST_PROMPTS.slice(0, 10);
export const VALIDATION_PROSE = PROSE_PROMPTS.slice(0, 10);
export const DEV_LIST = LIST_PROMPTS.slice(10);
export const DEV_PROSE = PROSE_PROMPTS.slice(10);

// ---------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------

export const VARIANTS = [
  {
    name: 'baseline_json',
    system:
      'You classify the user\'s prompt. Respond only with a JSON object of the form {"prompt_is_a_list": true} or {"prompt_is_a_list": false}. Answer true when the user is asking for content best rendered as a bulleted/numbered list (e.g., "list X", "give me N things", "steps to Y"). Answer false for prose, stories, explanations, or single-value answers.',
    prefill: '{"prompt_is_a_list": ',
    branches: ['true}', 'false}'],
    parse: (s) => s.startsWith('true'),
  },
  {
    name: 'inverted_is_prose',
    system:
      'You classify the user\'s prompt. Respond only with a JSON object of the form {"prompt_is_prose": true} or {"prompt_is_prose": false}. Answer true when the best response is prose/narrative (paragraph, story, explanation, single-value answer). Answer false when the best response is a bulleted or numbered list.',
    prefill: '{"prompt_is_prose": ',
    branches: ['true}', 'false}'],
    parse: (s) => !s.startsWith('true'),
  },
  {
    name: 'format_string',
    system:
      'You classify the user\'s prompt. Respond only with a JSON object: {"format": "list"} when the best response is a bulleted or numbered list, or {"format": "prose"} when the best response is narrative text.',
    prefill: '{"format": "',
    branches: ['list"}', 'prose"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'type_uppercase',
    system:
      'You classify the user\'s prompt. Respond only with {"type": "LIST"} (for bulleted or numbered lists) or {"type": "PROSE"} (for narrative text, stories, explanations, translations, single facts).',
    prefill: '{"type": "',
    branches: ['LIST"}', 'PROSE"}'],
    parse: (s) => s.startsWith('LIST'),
  },
  {
    name: 'natural_completion',
    system:
      'Classify the user\'s request by completing the sentence with a single word.',
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'paragraph.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'yes_no',
    system:
      'Should the user\'s request be answered as a bulleted list? Respond with exactly one word: yes or no.',
    prefill: 'Answer: ',
    branches: ['yes', 'no'],
    parse: (s) => s.toLowerCase().startsWith('y'),
  },
  {
    name: 'few_shot_json',
    system:
      'You classify the user\'s prompt. Examples:\n- "list five python libraries" → {"prompt_is_a_list": true}\n- "explain relativity" → {"prompt_is_a_list": false}\n- "name three cats" → {"prompt_is_a_list": true}\n- "write a haiku about autumn" → {"prompt_is_a_list": false}\n- "what are 5 tips for sleep" → {"prompt_is_a_list": true}\n- "tell me a story about a dragon" → {"prompt_is_a_list": false}\nRespond only with the JSON object.',
    prefill: '{"prompt_is_a_list": ',
    branches: ['true}', 'false}'],
    parse: (s) => s.startsWith('true'),
  },
  {
    name: 'strict_definition',
    system:
      'Classify the user\'s request. Respond {"prompt_is_a_list": true} only if the ideal answer is multiple discrete items (a bulleted or numbered list of 2+ items). Respond {"prompt_is_a_list": false} for everything else: single-value answers, prose, explanations, stories, poems, emails, factual Q&A, translations, descriptions of one thing.',
    prefill: '{"prompt_is_a_list": ',
    branches: ['true}', 'false}'],
    parse: (s) => s.startsWith('true'),
  },
  {
    name: 'short_json',
    system:
      'Output only JSON: {"list": true} or {"list": false}. true = answer should be a bulleted list. false = prose or a single answer.',
    prefill: '{"list": ',
    branches: ['true}', 'false}'],
    parse: (s) => s.startsWith('true'),
  },
  {
    name: 'binary_answer',
    system:
      'Output exactly one character: 1 if the user\'s request should be answered as a bulleted list, else 0.',
    prefill: 'Answer: ',
    branches: ['1', '0'],
    parse: (s) => s.trim() === '1',
  },
  {
    name: 'yes_no_with_rules',
    system:
      'A bulleted list is appropriate when the user asks to "list", "name N", "give N things", "ways to", "reasons for", "tips", "steps", "ingredients", "examples of", "what are the [plural]". Everything else (stories, poems, explanations of a concept, translations, single facts, emails, descriptions) should be prose. Respond yes if the prompt below should be answered as a bulleted list, else no.',
    prefill: 'Answer: ',
    branches: ['yes', 'no'],
    parse: (s) => s.toLowerCase().startsWith('y'),
  },
  // --- Round 2: focused on the top 3 winners, attacking their class-bias
  //     failure modes (format_string misses prose; natural_completion misses
  //     list; type_uppercase is balanced but low overall).
  {
    name: 'r2_format_fewshot_prose_heavy',
    system:
      `You classify the user's prompt. Respond only with a JSON object: {"format": "list"} when the best response is a bulleted/numbered list of discrete items, or {"format": "prose"} for everything else.

Examples:
- "list five python libraries" → {"format": "list"}
- "explain photosynthesis" → {"format": "prose"}
- "write a haiku about autumn" → {"format": "prose"}
- "give me three cat names" → {"format": "list"}
- "tell me a story about a cat" → {"format": "prose"}
- "what does serendipity mean?" → {"format": "prose"}
- "translate good morning to French" → {"format": "prose"}`,
    prefill: '{"format": "',
    branches: ['list"}', 'prose"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_format_prose_default',
    system:
      `Classify the user's prompt. Respond with exactly one JSON object. Default to {"format": "prose"}. Only choose {"format": "list"} when the user clearly asks for multiple discrete items — phrasings like "list N things", "name N X", "give me N reasons", "top N", "suggest some X", "steps to do X", "ingredients for Y", "types of Z", "what are the [plural noun]".`,
    prefill: '{"format": "',
    branches: ['list"}', 'prose"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_format_prose_first_grammar',
    system:
      `You classify the user's prompt. Respond only with a JSON object: {"format": "prose"} for prose/narrative responses, or {"format": "list"} for bulleted/numbered lists.`,
    prefill: '{"format": "',
    branches: ['prose"}', 'list"}'], // branch-order flipped (prose is first option)
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_format_items_text',
    system:
      `Classify the user's prompt. Respond only with {"format": "items"} when the best response is a bulleted/numbered list, or {"format": "text"} when it's narrative text, a story, an explanation, a translation, or a single-value fact.`,
    prefill: '{"format": "',
    branches: ['items"}', 'text"}'],
    parse: (s) => s.startsWith('items'),
  },
  {
    name: 'r2_format_bulleted_paragraph',
    system:
      `Classify the user's prompt. Respond only with {"format": "bulleted"} when the best response is a bulleted or numbered list, or {"format": "paragraph"} when it's paragraph-length prose, a story, an explanation, a translation, or a single-value fact.`,
    prefill: '{"format": "',
    branches: ['bulleted"}', 'paragraph"}'],
    parse: (s) => s.startsWith('bulleted'),
  },
  {
    name: 'r2_type_lowercase',
    system:
      `Classify the user's prompt. Respond only with {"type": "list"} (bulleted/numbered list) or {"type": "prose"} (narrative, story, explanation, single fact, translation).`,
    prefill: '{"type": "',
    branches: ['list"}', 'prose"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_natural_fewshot',
    system:
      `Classify the user's request by completing the sentence with one word. Examples:
- User: "list 5 fruits" → The user wants the answer as a list.
- User: "tell me a story" → The user wants the answer as a paragraph.
- User: "give me 3 tips" → The user wants the answer as a list.
- User: "explain relativity" → The user wants the answer as a paragraph.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'paragraph.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_natural_list_heavy',
    system:
      `Classify the user's request by completing the sentence. A list is for "list/name/give N things", "ways to", "tips for", "steps to", "reasons for", "examples of", "what are the [plural]". A paragraph is for stories, explanations, translations, poems, emails, descriptions, or single facts.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'paragraph.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r2_compact_format',
    system:
      `Classify the user's prompt. Output "Format: list" if the answer is best as a bulleted list, else "Format: prose".`,
    prefill: 'Format: ',
    branches: ['list', 'prose'],
    parse: (s) => s.trim().startsWith('list'),
  },
  {
    name: 'r2_intent_story_list',
    system:
      `Classify the user's intent. Complete the sentence. Use "list" when the user wants enumerated items ("list", "name N", "give N things", "ways to", "tips for", "steps", "reasons", "examples of"). Use "story" for everything else — prose, stories, explanations, translations, poems, emails, descriptions, single facts.`,
    prefill: 'The intent is to get a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// --- Round 3: 5 variations of each of the top-4 round-2 winners.

// A. Variations of r2_intent_story_list (round 2 winner, 83.8%, balanced).
//    "The intent is to get a list./story."
const R3_A_VARIANTS = [
  {
    name: 'r3_a1_intent_short',
    system: `Classify the user's intent. Complete the sentence.`,
    prefill: 'The intent is to get a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_a2_intent_fewshot',
    system:
      `Classify the user's intent. Complete the sentence. Examples:
- "list 5 fruits" → The intent is to get a list.
- "tell me a short story" → The intent is to get a story.
- "give me 3 tips for sleep" → The intent is to get a list.
- "explain photosynthesis" → The intent is to get a story.
- "name three jazz musicians" → The intent is to get a list.
- "write a limerick" → The intent is to get a story.`,
    prefill: 'The intent is to get a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_a3_reply_should_be',
    system:
      `Classify the user's intent. Complete the sentence. Use "list" for enumerated items (list/name N/give N/ways/tips/steps/reasons/examples). Use "story" for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_a4_intent_symmetric_rules',
    system:
      `Classify the user's intent. "List" answers work for "list N", "name N", "give N things", "ways to", "tips", "steps", "reasons", "examples of", "what are the plural Xs". "Story" answers work for "tell me about", "explain", "describe", "translate", "write a story/poem/email/haiku/essay", "who was/what is/when did", "summarize".`,
    prefill: 'The intent is to get a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_a5_intent_essay_vocab',
    system:
      `Classify the user's intent. Use "list" when the user wants multiple discrete items ("list", "name N", "give N things", "ways to", "tips", "steps", "reasons", "examples"). Use "essay" for everything else — prose, stories, explanations, translations, poems, emails, descriptions, factual Q&A.`,
    prefill: 'The intent is to get an ',
    branches: ['list.', 'essay.'],
    parse: (s) => s.startsWith('list'),
  },
];

// B. Variations of r2_format_bulleted_paragraph (78.8%, list-biased 39/24).
const R3_B_VARIANTS = [
  {
    name: 'r3_b1_bulleted_fewshot_prose_heavy',
    system:
      `You classify the user's prompt. Respond only with a JSON object: {"format": "bulleted"} for bulleted/numbered lists, or {"format": "paragraph"} for everything else (prose, stories, explanations, translations, single facts, poems, emails, descriptions).

Examples:
- "list 5 fruits" → {"format": "bulleted"}
- "tell me a story about a cat" → {"format": "paragraph"}
- "explain photosynthesis" → {"format": "paragraph"}
- "give me 3 tips" → {"format": "bulleted"}
- "translate good morning" → {"format": "paragraph"}
- "what does X mean" → {"format": "paragraph"}
- "describe a sunset" → {"format": "paragraph"}
- "name four cats" → {"format": "bulleted"}`,
    prefill: '{"format": "',
    branches: ['bulleted"}', 'paragraph"}'],
    parse: (s) => s.startsWith('bulleted'),
  },
  {
    name: 'r3_b2_bulleted_default_paragraph',
    system:
      `Classify the user's prompt. Respond with exactly one JSON object. Default to {"format": "paragraph"}. Only choose {"format": "bulleted"} if the user clearly asks for multiple discrete items — phrasings like "list N", "name N", "give N", "top N", "suggest some", "steps", "ingredients", "types of", "what are the plural Xs".`,
    prefill: '{"format": "',
    branches: ['bulleted"}', 'paragraph"}'],
    parse: (s) => s.startsWith('bulleted'),
  },
  {
    name: 'r3_b3_format_bulleted_story',
    system:
      `Classify the user's prompt. Respond only with {"format": "bulleted"} (bulleted/numbered list) or {"format": "story"} (narrative, story, explanation, translation, single fact).`,
    prefill: '{"format": "',
    branches: ['bulleted"}', 'story"}'],
    parse: (s) => s.startsWith('bulleted'),
  },
  {
    name: 'r3_b4_bulleted_prose_vocab',
    system:
      `Classify the user's prompt. Respond only with {"format": "bullets"} for bulleted/numbered lists or {"format": "prose"} for narrative text, stories, explanations, translations, single facts.`,
    prefill: '{"format": "',
    branches: ['bullets"}', 'prose"}'],
    parse: (s) => s.startsWith('bullets'),
  },
  {
    name: 'r3_b5_itemized_vs_paragraph',
    system:
      `Classify the user's prompt. Respond only with {"format": "itemized"} for enumerated/bulleted/numbered lists or {"format": "paragraph"} for stories, explanations, translations, poems, single facts, or any narrative text.`,
    prefill: '{"format": "',
    branches: ['itemized"}', 'paragraph"}'],
    parse: (s) => s.startsWith('itemized'),
  },
];

// C. Variations of r2_format_fewshot_prose_heavy (76.3%, list-biased 40/21).
const R3_C_VARIANTS = [
  {
    name: 'r3_c1_fewshot_list_story_vocab',
    system:
      `Classify the user's prompt. Respond only with {"format": "list"} or {"format": "story"}. Examples:
- "list five python libraries" → {"format": "list"}
- "explain photosynthesis" → {"format": "story"}
- "write a haiku about autumn" → {"format": "story"}
- "give me three cat names" → {"format": "list"}
- "tell me about the printing press" → {"format": "story"}
- "what does serendipity mean?" → {"format": "story"}`,
    prefill: '{"format": "',
    branches: ['list"}', 'story"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_c2_fewshot_more_prose_examples',
    system:
      `Classify the user's prompt. Respond only with a JSON object: {"format": "list"} for bulleted/numbered lists, or {"format": "prose"} for everything else.

Examples:
- "list five python libraries" → {"format": "list"}
- "give me three cat names" → {"format": "list"}
- "explain photosynthesis" → {"format": "prose"}
- "write a haiku about autumn" → {"format": "prose"}
- "tell me about the printing press" → {"format": "prose"}
- "translate good morning to French" → {"format": "prose"}
- "what does serendipity mean?" → {"format": "prose"}
- "describe a sunset over the ocean" → {"format": "prose"}
- "write a professional email" → {"format": "prose"}
- "who wrote Hamlet?" → {"format": "prose"}`,
    prefill: '{"format": "',
    branches: ['list"}', 'prose"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_c3_fewshot_balanced_5_5',
    system:
      `Classify the user's prompt. Respond only with {"format": "list"} or {"format": "story"}. Examples:
- "list 5 fruits" → {"format": "list"}
- "name three jazz musicians" → {"format": "list"}
- "give me 3 tips for sleep" → {"format": "list"}
- "what are the planets?" → {"format": "list"}
- "top 10 songs from the 80s" → {"format": "list"}
- "tell me a story about a dragon" → {"format": "story"}
- "write a haiku about autumn" → {"format": "story"}
- "explain how blockchain works" → {"format": "story"}
- "who was Marie Curie?" → {"format": "story"}
- "describe the feeling of nostalgia" → {"format": "story"}`,
    prefill: '{"format": "',
    branches: ['list"}', 'story"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_c4_fewshot_edge_cases',
    system:
      `Classify the user's prompt. Respond only with {"format": "list"} (bulleted/numbered list of discrete items) or {"format": "story"} (narrative/prose). Edge cases:
- "what are the symptoms of X" → {"format": "list"}  (discrete symptoms)
- "what is photosynthesis" → {"format": "story"}  (one concept explained)
- "write a haiku" → {"format": "story"}  (one poem, narrative form)
- "name five cats" → {"format": "list"}
- "summarize Pride and Prejudice" → {"format": "story"}  (one summary)
- "suggest gift ideas" → {"format": "list"}  (multiple ideas)`,
    prefill: '{"format": "',
    branches: ['list"}', 'story"}'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_c5_fewshot_intent_hybrid',
    system:
      `Classify the user's intent. Use "list" when the user wants multiple discrete items. Use "story" for everything else — prose, narrative, explanations, translations, single facts, poems, emails.

Examples:
- "list five fruits" → The intent is to get a list.
- "write a haiku about autumn" → The intent is to get a story.
- "give me 3 tips" → The intent is to get a list.
- "tell me about Marie Curie" → The intent is to get a story.
- "what are the planets?" → The intent is to get a list.
- "explain blockchain" → The intent is to get a story.`,
    prefill: 'The intent is to get a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// D. Variations of r2_natural_list_heavy (75%, prose-biased 20/40).
const R3_D_VARIANTS = [
  {
    name: 'r3_d1_natural_story_vocab',
    system:
      `Classify the user's request by completing the sentence. A list is for "list/name/give N things", "ways to", "tips for", "steps to", "reasons for", "examples of". A story is for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_d2_natural_simplified',
    system:
      `Classify the user's request.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_d3_response_format',
    system:
      `Classify the user's request by completing the sentence. Use "list" when the answer should be multiple items (list/name N/give N/ways/tips/steps/reasons/examples). Use "story" for narrative answers (stories, explanations, translations, poems, single facts).`,
    prefill: 'The response format should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_d4_natural_list_default',
    system:
      `Classify the user's request. Default to "list". Only use "story" when the user clearly asks for narrative: "tell me a story", "write a poem/haiku/email", "describe X", "explain X", "translate X", "what does X mean", "who was/what is/when did".`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r3_d5_answer_should_be',
    system:
      `Classify the user's request by completing the sentence.`,
    prefill: 'The answer should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// Append all round 3 variants to the main list.
VARIANTS.push(...R3_A_VARIANTS, ...R3_B_VARIANTS, ...R3_C_VARIANTS, ...R3_D_VARIANTS);

// --- Round 4: 5 variations of each of the top-5 *clean* (leak-free) round-3
// variants, plus a handful of combinations. Few-shot examples were hand-picked
// to not overlap with LIST_PROMPTS / PROSE_PROMPTS (even by substring).
// Novel list examples: camping, clouds, board games, bridges, morning habits.
// Novel prose examples: time-traveling cat, refrigerator, sound of rain,
// lightbulb invention, Spanish greeting.

// A. Variations of r3_d4_natural_list_default (97.5% / 95%)
const R4_A_VARIANTS = [
  {
    name: 'r4_a1_d4_plus_fewshot',
    system:
      `Classify the user's request. Default to "list". Only use "story" when the user clearly asks for narrative: "tell me a story", "write a poem/haiku/email", "describe X", "explain X", "translate X", "what does X mean", "who was/what is/when did".

Examples:
- "what to pack for a camping trip" → The user wants the answer as a list.
- "write a story about a time-traveling cat" → The user wants the answer as a story.
- "types of clouds" → The user wants the answer as a list.
- "explain how a refrigerator cools things" → The user wants the answer as a story.
- "most popular board games" → The user wants the answer as a list.
- "describe the sound of rain" → The user wants the answer as a story.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_a2_d4_shorter_rules',
    system:
      `Classify the user's request. Default to "list". Use "story" only for narrative (stories, poems, descriptions, explanations, translations, single facts).`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_a3_d4_extended_triggers',
    system:
      `Classify the user's request. Default to "list". Use "story" only when the user asks for narrative/prose: "tell me a story", "write a poem/haiku/limerick/email/essay/letter", "describe", "explain", "translate", "summarize", "what does X mean", "who was/is", "what is X", "when did", "why does", "how does (concept)", "compose".`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_a4_d4_reply_prefill',
    system:
      `Classify the user's request. Default to "list". Use "story" only for narrative: "tell me a story", "write a poem/haiku/email", "describe X", "explain X", "translate X", "what does X mean", "who was/what is/when did".`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_a5_d4_inline_examples',
    system:
      `Classify the user's request. Default to "list" (covers "list N", "name N", "give N things", "ways to", "tips for", "steps to", "reasons for", "examples of", "suggest some", "top N", "what are the [plural]"). Use "story" ONLY for narrative requests ("tell me a story", "write a poem/haiku/email", "describe X", "explain X", "translate X", "what does X mean", "who was/what is/when did", "summarize", "compose a toast").`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// B. Variations of r3_d2_natural_simplified (95% dev, minimal system)
const R4_B_VARIANTS = [
  {
    name: 'r4_b1_d2_plus_fewshot',
    system:
      `Classify the user's request.

Examples:
- "what to pack for a camping trip" → The user wants the answer as a list.
- "write a story about a time-traveling cat" → The user wants the answer as a story.
- "types of clouds" → The user wants the answer as a list.
- "explain how a refrigerator cools things" → The user wants the answer as a story.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_b2_d2_one_sentence_rule',
    system:
      `Classify the user's request. Use "list" for requests that want multiple discrete items, "story" for narrative, explanation, or a single-value answer.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_b3_d2_reply_prefill',
    system: `Classify the user's request.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_b4_d2_classify_word',
    system: `Classify.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_b5_d2_no_system',
    system: `You read requests and say whether they'd be better answered as a list or a story.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// C. Variations of r3_a3_reply_should_be (95% / 90%)
const R4_C_VARIANTS = [
  {
    name: 'r4_c1_a3_plus_default_list',
    system:
      `Classify the user's intent. Default to "list". Use "story" only for narrative answers — stories, poems, explanations of a concept, translations, single-value facts, descriptions, emails.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_c2_a3_plus_fewshot',
    system:
      `Classify the user's intent. Use "list" for enumerated items, "story" for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.

Examples:
- "what to pack for a camping trip" → The reply should be a list.
- "write a story about a time-traveling cat" → The reply should be a story.
- "habits of a successful morning" → The reply should be a list.
- "who invented the lightbulb" → The reply should be a story.
- "famous bridges" → The reply should be a list.
- "translate hello to Spanish" → The reply should be a story.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_c3_a3_output_prefill',
    system:
      `Classify the user's intent. "List" answers work for "list N", "name N", "give N things", "ways to", "tips", "steps", "reasons", "examples of". "Story" answers work for narrative, explanations, translations, single facts, poems, emails, descriptions.`,
    prefill: 'The output should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_c4_a3_shorter_rules',
    system:
      `Classify the user's intent. "List" for multiple discrete items. "Story" for narrative, prose, or single facts.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_c5_a3_default_story',
    system:
      `Classify the user's intent. Default to "story". Use "list" only if the user clearly asks for multiple discrete items: "list N", "name N", "give N", "top N", "suggest some", "ways to", "tips", "steps", "reasons", "examples of".`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// D. Variations of r3_d1_natural_story_vocab (93.8%)
const R4_D_VARIANTS = [
  {
    name: 'r4_d1_combine_d4_rules',
    system:
      `Classify the user's request by completing the sentence. Default to "list". Use "story" only when the user clearly asks for narrative, a poem, an explanation of a concept, a translation, a description, an email, or a single-value factual answer.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_d2_plus_novel_fewshot',
    system:
      `Classify the user's request. A list is for "list/name/give N things", "ways to", "tips for", "steps to", "reasons for", "examples of". A story is for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.

Examples:
- "what to pack for a camping trip" → The user wants the answer as a list.
- "write a story about a time-traveling cat" → The user wants the answer as a story.
- "most popular board games" → The user wants the answer as a list.
- "describe the sound of rain" → The user wants the answer as a story.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_d3_story_defined_negatively',
    system:
      `Classify the user's request. Use "story" when the user asks for prose/narrative/a single fact: "tell me a story", "write a poem/haiku/email", "describe", "explain", "translate", "summarize", "what does X mean", "who was", "what is", "when did". Use "list" for everything else.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_d4_shorter_rules',
    system:
      `Classify the user's request. "List" for enumerated items. "Story" for narrative, explanations, single facts.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_d5_best_format_prefill',
    system:
      `Classify the user's request. A list is for "list/name/give N things", "ways to", "tips for", "steps to", "reasons for", "examples of". A story is for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.`,
    prefill: 'The best format is a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

// E. Combinations — pulling elements from multiple winners.
const R4_E_VARIANTS = [
  {
    name: 'r4_e1_d4rules_a3prefill_fewshot',
    system:
      `Classify the user's intent. Default to "list". Use "story" only for narrative: "tell me a story", "write a poem/haiku/email", "describe X", "explain X", "translate X", "what does X mean", "who was/what is/when did".

Examples:
- "what to pack for a camping trip" → The reply should be a list.
- "write a story about a time-traveling cat" → The reply should be a story.
- "habits of a successful morning" → The reply should be a list.
- "who invented the lightbulb" → The reply should be a story.`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_e2_d4rules_3shot_novel',
    system:
      `Classify the user's request. Default to "list". Use "story" only for narrative, a poem, an explanation of a concept, a translation, a description, an email, or a single-value factual answer.

Examples:
- "types of clouds" → The user wants the answer as a list.
- "explain how a refrigerator cools things" → The user wants the answer as a story.
- "famous bridges" → The user wants the answer as a list.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_e3_symmetric_rules_reply_prefill',
    system:
      `Classify the user's intent. "List" answers work for "list N", "name N", "give N things", "ways to", "tips", "steps", "reasons", "examples of", "what are the [plural Xs]". "Story" answers work for "tell me about", "explain", "describe", "translate", "write a story/poem/email/haiku/essay", "who was/what is/when did", "summarize".`,
    prefill: 'The reply should be a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_e4_d1_rules_simple_prefill',
    system:
      `Classify the user's request. A list is for "list/name/give N things", "ways to", "tips for", "steps to", "reasons for", "examples of". A story is for everything else — stories, explanations, translations, poems, emails, descriptions, single facts.`,
    prefill: 'The answer is a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
  {
    name: 'r4_e5_minimal_plus_3shot',
    system:
      `Classify the user's request.

Examples:
- "things to pack for a camping trip" → The user wants the answer as a list.
- "write a story about a time-traveling cat" → The user wants the answer as a story.
- "types of clouds" → The user wants the answer as a list.`,
    prefill: 'The user wants the answer as a ',
    branches: ['list.', 'story.'],
    parse: (s) => s.startsWith('list'),
  },
];

VARIANTS.push(...R4_A_VARIANTS, ...R4_B_VARIANTS, ...R4_C_VARIANTS, ...R4_D_VARIANTS, ...R4_E_VARIANTS);

// Helper: filter variants by name prefix (for running just round-2).
export function variantsWithPrefix(prefix) {
  return VARIANTS.filter((v) => v.name.startsWith(prefix));
}

export async function runVariantsOnDev(ctx, variants, onProgress) {
  const dev = makeLabelled(DEV_LIST, DEV_PROSE);
  const summaries = [];
  for (let v = 0; v < variants.length; v++) {
    const variant = variants[v];
    const t0 = performance.now();
    const res = await runVariantOn(ctx, variant, dev, (p) =>
      onProgress?.({ variantIdx: v, variantName: variant.name, variantTotal: variants.length, ...p })
    );
    res.wallMs = performance.now() - t0;
    console.log(`[eval] ${variant.name}: ${res.correct}/${res.total} = ${(res.accuracy * 100).toFixed(1)}% (${res.wallMs.toFixed(0)} ms)`);
    summaries.push(res);
  }
  return summaries;
}

// ---------------------------------------------------------------------------
// Core: one classifier call
// ---------------------------------------------------------------------------

export async function classify(ctx, variant, userPrompt) {
  const messages = [
    { role: 'system', content: variant.system },
    { role: 'user', content: userPrompt },
  ];
  const templated = ctx.generator.tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });
  const fullText = templated + variant.prefill;

  const grammar = unionGrammars(variant.branches.map((b) => compileLiteral(b)));
  const processor = new GrammarLogitsProcessor({
    grammar,
    tokenizer: ctx.generator.tokenizer,
    tokenText: ctx.tokenText,
    eosTokenIds: ctx.eosTokenIds,
  });
  const processors = new LogitsProcessorList();
  processors.push(processor);

  let result = '';
  const streamer = new TextStreamer(ctx.generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => { result += t; },
  });

  // Longest branch char count + a little slack. Most branches are <= 6 chars;
  // 16 tokens gives plenty of room even for multi-token decompositions.
  await ctx.generator(fullText, {
    max_new_tokens: 16,
    do_sample: false,
    logits_processor: processors,
    streamer,
    return_full_text: false,
  });

  return { prediction: variant.parse(result), raw: result };
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

export async function runVariantOn(ctx, variant, labelledPrompts, onProgress) {
  const out = [];
  for (let i = 0; i < labelledPrompts.length; i++) {
    const { prompt, expected } = labelledPrompts[i];
    const { prediction, raw } = await classify(ctx, variant, prompt);
    out.push({ prompt, expected, prediction, raw, correct: prediction === expected });
    if (onProgress) onProgress({ done: i + 1, total: labelledPrompts.length, last: out[out.length - 1] });
  }
  const correct = out.filter((r) => r.correct).length;
  return { variant: variant.name, accuracy: correct / out.length, correct, total: out.length, results: out };
}

export function makeLabelled(listPrompts, prosePrompts) {
  return [
    ...listPrompts.map((prompt) => ({ prompt, expected: true })),
    ...prosePrompts.map((prompt) => ({ prompt, expected: false })),
  ];
}

// Run every variant over the dev set; returns a summary plus per-variant details.
export async function runAllOnDev(ctx, onProgress) {
  const dev = makeLabelled(DEV_LIST, DEV_PROSE);
  const summaries = [];
  for (let v = 0; v < VARIANTS.length; v++) {
    const variant = VARIANTS[v];
    const t0 = performance.now();
    const res = await runVariantOn(ctx, variant, dev, (p) =>
      onProgress?.({ variantIdx: v, variantName: variant.name, variantTotal: VARIANTS.length, ...p })
    );
    res.wallMs = performance.now() - t0;
    console.log(`[eval] ${variant.name}: ${res.correct}/${res.total} = ${(res.accuracy * 100).toFixed(1)}% (${res.wallMs.toFixed(0)} ms)`);
    summaries.push(res);
  }
  return summaries;
}

export async function runOnValidation(ctx, variantNames, onProgress) {
  const val = makeLabelled(VALIDATION_LIST, VALIDATION_PROSE);
  const wanted = new Set(variantNames);
  const summaries = [];
  for (const variant of VARIANTS.filter((v) => wanted.has(v.name))) {
    const res = await runVariantOn(ctx, variant, val, onProgress);
    console.log(`[eval:val] ${variant.name}: ${res.correct}/${res.total} = ${(res.accuracy * 100).toFixed(1)}%`);
    summaries.push(res);
  }
  return summaries;
}
