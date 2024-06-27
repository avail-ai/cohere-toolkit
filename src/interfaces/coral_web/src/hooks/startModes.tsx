import { DEFAULT_CHAT_TOOL } from '@/cohere-client';
import { IconName } from '@/components/Shared';
import { useParamsStore } from '@/stores';
import { ConfigurableParams } from '@/stores/slices/paramsSlice';

export enum StartMode {
  UNGROUNDED = 'ungrounded',
  WEB_SEARCH = 'web_search',
  TOOLS = 'tools',
}

type Prompt = {
  title: string;
  description: React.ReactNode;
  icon: IconName;
  prompt: string;
};

type Mode = {
  id: StartMode;
  title: string;
  description: string;
  params: Partial<ConfigurableParams>;
  promptOptions: Prompt[];
  onChange?: VoidFunction;
};

const UNGROUNDED_PROMPTS: Prompt[] = [
  {
    title: 'Acquiring a commercial property',
    description: (
      <>
        What are the legal steps involved in acquiring a commercial property through a share purchase?
      </>
    ),
    icon: 'globe-stand',
    prompt:
      'What are the legal steps involved in acquiring a commercial property through a share purchase?',
  },
  {
    title: 'Minority shareholders',
    description: 'What should be included in a shareholders\' agreement to protect minority shareholders?',
    icon: 'globe-stand',
    prompt:
      'What should be included in a shareholders\' agreement to protect minority shareholders?',
  },
  {
    title: 'Non-compete clauses',
    description: 'What are the legal requirements for drafting a non-compete clause in an employment contract?',
    icon: 'code',
    prompt: `What are the legal requirements for drafting a non-compete clause in an employment contract?`,
  },
];

const COMMAND_PROMPTS: Prompt[] = [
  {
    title: 'Letter of intent',
    description: 'Drafting a letter',
    icon: 'newspaper',
    prompt: 'Draft a letter of intent for the acquisition of a commercial property.',
  },
  {
    title: 'Shareholder\'s Agreement',
    description: 'Generate a template',
    icon: 'flask',
    prompt: 'Generate a template for a shareholders\' agreement for a private limited company.',
  },
  {
    title: 'Redundancy policy',
    description: 'Generate a policy template',
    icon: 'book',
    prompt: `Generate a redundancy policy template compliant with UK employment laws.`,
  },
];

export const useStartModes = () => {
  const { params } = useParamsStore();

  const modes: Mode[] = [
    {
      id: StartMode.UNGROUNDED,
      title: 'Ask Questions',
      description: 'Use this chat bot without any access to external sources to answer questions',
      params: { fileIds: [], tools: [] },
      promptOptions: UNGROUNDED_PROMPTS,
    },
    {
      id: StartMode.UNGROUNDED,
      title: 'Give Commands',
      description: 'Ask the chat bot to draft/generate documents/letters/emails for you',
      params: { fileIds: [], tools: [] },
      promptOptions: COMMAND_PROMPTS,
    },
  ];

  const getSelectedModeIndex = (): number => {
    let selectedTabKey = StartMode.UNGROUNDED;
    if (params.tools && params.tools.length > 0) {
      selectedTabKey = StartMode.WEB_SEARCH;
    }
    return modes.findIndex((m) => m.id === selectedTabKey);
  };

  return { modes, getSelectedModeIndex };
};
