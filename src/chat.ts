import { OpenAI, AzureOpenAI } from 'openai';

const reasoningEfforts = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const;

type ReasoningEffort = typeof reasoningEfforts[number];

const isReasoningEffort = (effort: string): effort is ReasoningEffort =>
  (reasoningEfforts as readonly string[]).includes(effort);

export interface Review {
  lgtm: boolean;
  review_comment: string;
  hunk_header?: string;
}

export class InvalidReviewResponseError extends Error {
  constructor(reason: string) {
    super(`Invalid model review response: ${reason}.`);
    this.name = 'InvalidReviewResponseError';
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isReview = (value: unknown): value is Review => {
  if (
    !isRecord(value) ||
    typeof value.lgtm !== 'boolean' ||
    typeof value.review_comment !== 'string' ||
    (value.hunk_header !== undefined && typeof value.hunk_header !== 'string')
  ) {
    return false;
  }

  // A finding needs an explanation; an LGTM must not hide one from the caller.
  return value.lgtm
    ? value.review_comment.trim().length === 0
    : value.review_comment.trim().length > 0;
};

const parseReviewResponse = (content: string): Review | Review[] => {
  let json: unknown;
  try {
    json = JSON.parse(content);
  } catch {
    throw new InvalidReviewResponseError('expected valid JSON');
  }

  if (isRecord(json) && 'reviews' in json) {
    const reviews = json.reviews;
    if (Array.isArray(reviews) && reviews.every(isReview)) {
      // An explicit empty array means no findings, unlike an empty API response.
      return reviews;
    }
  } else if (isReview(json)) {
    // Keep compatibility with the original single-review response format.
    return json;
  }

  throw new InvalidReviewResponseError('expected a review or a reviews array with valid fields');
};

export class Chat {
  private openai: OpenAI | AzureOpenAI;
  private isAzure: boolean;
  private isGithubModels: boolean;

  private reasoningModels = ['o1', 'o1-2024-12-17', 'o1-mini', 'o1-mini-2024-09-12'];
  private reasoningPrefixes = ['o3', 'o4', 'gpt-5'];

  constructor(apikey: string) {
    this.isAzure = Boolean(
        process.env.AZURE_API_VERSION && process.env.AZURE_DEPLOYMENT,
    );

    this.isGithubModels = process.env.USE_GITHUB_MODELS === 'true';

    if (this.isAzure) {
      // Azure OpenAI configuration
      this.openai = new AzureOpenAI({
        apiKey: apikey,
        endpoint: process.env.OPENAI_API_ENDPOINT || '',
        apiVersion: process.env.AZURE_API_VERSION || '',
        deployment: process.env.AZURE_DEPLOYMENT || '',
      });
    } else {
      // Standard OpenAI configuration
      this.openai = new OpenAI({
        apiKey: apikey,
        baseURL: this.isGithubModels ? 'https://models.github.ai/inference' : process.env.OPENAI_API_ENDPOINT || 'https://api.openai.com/v1',
      });
    }
  }

  private get model(): string {
    return process.env.MODEL || (this.isGithubModels ? 'openai/gpt-4o-mini' : 'gpt-4o-mini');
  }

  private get normalizedModel(): string {
    return this.model.split('/').pop() || this.model;
  }

  private get isReasoningModel(): boolean {
    const model = this.normalizedModel.toLowerCase();
    return this.reasoningModels.includes(model) || this.reasoningPrefixes.some(prefix => model.startsWith(prefix));
  }

  private get reasoningEffortOption(): { reasoning_effort?: ReasoningEffort } {
    const effort = process.env.REASONING_EFFORT;
    if (!effort || !this.isReasoningModel) return {};
    if (!isReasoningEffort(effort)) {
      console.warn(`REASONING_EFFORT="${effort}" is invalid, ignoring. Valid values: ${reasoningEfforts.join(', ')}`);
      return {};
    }
    return { reasoning_effort: effort };
  }

  private generatePrompt = (patch: string) => {
    const answerLanguage = process.env.LANGUAGE
        ? `Answer me in ${process.env.LANGUAGE},`
        : '';

    const userPrompt = process.env.PROMPT || 'Please review the following code patch. Focus on potential bugs, risks, and improvement suggestions.';
    
    const jsonFormatRequirement = '\nProvide your feedback in a strict JSON format with the following structure:\n' +
        '{\n' +
        '  "reviews": [\n' +
        '    {\n' +
        '      "hunk_header": string, // The @@ hunk header (e.g., "@@ -10,5 +10,7 @@"), optional\n' +
        '      "lgtm": boolean, // true if this hunk looks good, false if there are concerns\n' +
        '      "review_comment": string // Your detailed review comments for this hunk. Can use markdown syntax. Empty string if lgtm is true.\n' +
        '    }\n' +
        '  ]\n' +
        '}\n' +
        'Review each hunk (marked by @@) separately and provide feedback for hunks that need improvement.\n' +
        'If no hunks need improvement, return {"reviews": []}.\n' +
        'Ensure your response is a valid JSON object with a reviews array.\n';

    return `${userPrompt}${jsonFormatRequirement} ${answerLanguage}:
    ${patch}
    `;
  };

  public codeReview = async (patch: string): Promise<Review[] | Review> => {
    if (!patch) {
      return {
        lgtm: true,
        review_comment: ""
      };
    }

    console.time('code-review cost');
    const prompt = this.generatePrompt(patch);

    const isReasoning = this.isReasoningModel;

    const res = await this.openai.chat.completions.create({
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      model: this.model,
      ...(isReasoning ? {} : {
        temperature: +(process.env.temperature || 0) || 1,
        top_p: +(process.env.top_p || 0) || 1,
      }),
      max_tokens: process.env.max_tokens ? +process.env.max_tokens : undefined,
      ...this.reasoningEffortOption,
      response_format: { type: "json_object" },
    }).finally(() => console.timeEnd('code-review cost'));

    const choice = res.choices?.[0];
    if (!choice?.message) {
      throw new InvalidReviewResponseError('no completion choice');
    }
    if (choice.finish_reason !== 'stop' || choice.message.refusal) {
      throw new InvalidReviewResponseError('generation did not finish normally');
    }
    const content = choice.message.content;
    if (typeof content !== 'string' || !content.trim()) {
      throw new InvalidReviewResponseError('empty completion content');
    }

    return parseReviewResponse(content);
  };
}
