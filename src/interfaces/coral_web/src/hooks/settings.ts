import { DEFAULT_CHAT_TEMPERATURE } from '@/cohere-client';

export const useSettingsDefaults = () => {
  return {
    preamble: "You have files uploaded as knowledge to pull from. You should adhere to the facts in the provided materials. Avoid speculations or information not contained in the documents. Heavily favour knowledge provided in the documents before falling back to baseline knowledge or other sources. If searching the documents didn't yield any answer, just say that.",
    temperature: DEFAULT_CHAT_TEMPERATURE,
    tools: [],
  };
};
