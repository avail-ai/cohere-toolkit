import { Transition } from '@headlessui/react';
import { NextPage } from 'next';
import { useContext, useEffect, useRef } from 'react';

import { Document, ManagedTool } from '@/cohere-client';
import { AgentsList } from '@/components/Agents/AgentsList';
import { ConnectDataModal } from '@/components/ConnectDataModal';
import Conversation from '@/components/Conversation';
import { ConversationError } from '@/components/ConversationError';
import ConversationListPanel from '@/components/ConversationList/ConversationListPanel';
import { AgentsLayout, Layout, LeftSection, MainSection } from '@/components/Layout';
import { ProtectedPage } from '@/components/ProtectedPage';
import { Spinner } from '@/components/Shared';
import { TOOL_PYTHON_INTERPRETER_ID } from '@/constants';
import { BannerContext } from '@/context/BannerContext';
import { ModalContext } from '@/context/ModalContext';
import { useAgent, useDefaultAgent } from '@/hooks/agents';
import { useIsDesktop } from '@/hooks/breakpoint';
import { useConversation } from '@/hooks/conversation';
import { useListAllDeployments } from '@/hooks/deployments';
import { useExperimentalFeatures } from '@/hooks/experimentalFeatures';
import { useSlugRoutes } from '@/hooks/slugRoutes';
import { useListTools, useShowUnauthedToolsModal } from '@/hooks/tools';
import {
  useCitationsStore,
  useConversationStore,
  useParamsStore,
  useSettingsStore,
} from '@/stores';
import { OutputFiles } from '@/stores/slices/citationsSlice';
import { cn, createStartEndKey, mapHistoryToMessages } from '@/utils';
import { parsePythonInterpreterToolFields } from '@/utils/tools';
import { useFileActions } from '@/hooks/files';

const Page: NextPage = () => {
  const { agentId, conversationId } = useSlugRoutes();

  const prevConversationId = useRef<string>();
  const { setConversation } = useConversationStore();
  const {
    settings: { isConvListPanelOpen, isMobileConvListPanelOpen },
  } = useSettingsStore();

  const isDesktop = useIsDesktop();
  const isMobile = !isDesktop;

  const { addCitation, resetCitations, saveOutputFiles } = useCitationsStore();
  const { clearComposerFiles } = useFileActions();
  const {
    params: { deployment },
    setParams,
    resetFileParams,
  } = useParamsStore();
  const { show: showUnauthedToolsModal, onDismissed } = useShowUnauthedToolsModal();
  const { data: allDeployments } = useListAllDeployments();
  const { data: agent } = useAgent({ agentId });
  useDefaultAgent(!agentId);
  const { data: tools } = useListTools();
  const { data: experimentalFeatures } = useExperimentalFeatures();
  const isLangchainModeOn = !!experimentalFeatures?.USE_EXPERIMENTAL_LANGCHAIN;
  const isAgentsModeOn = !!experimentalFeatures?.USE_AGENTS_VIEW;

  const { setMessage } = useContext(BannerContext);
  const { open, close } = useContext(ModalContext);

  const {
    data: conversation,
    isLoading,
    isError,
    error,
  } = useConversation({
    conversationId: conversationId,
  });

  useEffect(() => {
    if (showUnauthedToolsModal) {
      open({
        title: 'Connect your data',
        content: (
          <ConnectDataModal
            onClose={() => {
              onDismissed();
              close();
            }}
          />
        ),
      });
    }
  }, [showUnauthedToolsModal]);

  useEffect(() => {
    if (conversationId && prevConversationId.current && prevConversationId.current !== conversationId) {
      resetCitations();
      resetFileParams();
      clearComposerFiles();
    }
    prevConversationId.current = conversationId;

    const agentTools = (agent?.tools
      .map((name) => (tools ?? [])?.find((t) => t.name === name))
      .filter((t) => t !== undefined) ?? []) as ManagedTool[];
    setParams({
      tools: agentTools,
    });
  }, [conversationId, resetCitations, agent, tools]);

  useEffect(() => {
    if (!conversation) return;

    const messages = mapHistoryToMessages(
      conversation?.messages?.sort((a, b) => a.position - b.position)
    );

    setConversation({ name: conversation.title, messages });

    let documentsMap: { [documentId: string]: Document } = {};
    let outputFilesMap: OutputFiles = {};

    (conversation?.messages ?? []).forEach((message) => {
      message.documents?.forEach((doc) => {
        const docId = doc.document_id ?? '';
        documentsMap[docId] = doc;

        const toolName = (doc.tool_name ?? '').toLowerCase();

        if (toolName === TOOL_PYTHON_INTERPRETER_ID) {
          const { outputFile } = parsePythonInterpreterToolFields(doc);

          if (outputFile) {
            outputFilesMap[outputFile.filename] = {
              name: outputFile.filename,
              data: outputFile.b64_data,
              documentId: docId,
            };
          }
        }
      });
      message.citations?.forEach((citation) => {
        const startEndKey = createStartEndKey(citation.start ?? 0, citation.end ?? 0);
        const documents = citation.document_ids?.map((id) => documentsMap[id]) ?? [];
        addCitation(message.generation_id ?? '', startEndKey, documents);
      });
    });

    saveOutputFiles(outputFilesMap);
  }, [conversation?.id, setConversation]);

  useEffect(() => {
    if (!deployment && allDeployments) {
      const firstAvailableDeployment = allDeployments.find((d) => d.is_available);
      if (firstAvailableDeployment) {
        setParams({ deployment: firstAvailableDeployment.name });
      }
    }
  }, [deployment, allDeployments]);

  useEffect(() => {
    if (!isLangchainModeOn) return;
    setMessage('You are using an experimental langchain multihop flow. There will be bugs.');
  }, [isLangchainModeOn]);

  if (isAgentsModeOn) {
    return (
      <ProtectedPage>
        <AgentsLayout showSettingsDrawer>
          <LeftSection>
            <AgentsList />
          </LeftSection>
          <MainSection>
            <div className="flex h-full">
              <Transition
                as="section"
                appear
                show={(isMobileConvListPanelOpen && isMobile) || (isConvListPanelOpen && isDesktop)}
                enterFrom="translate-x-full lg:translate-x-0 lg:min-w-0 lg:max-w-0"
                enterTo="translate-x-0 lg:min-w-[300px] lg:max-w-[300px]"
                leaveFrom="translate-x-0 lg:min-w-[300px] lg:max-w-[300px]"
                leaveTo="translate-x-full lg:translate-x-0 lg:min-w-0 lg:max-w-0"
                className={cn(
                  'z-main-section flex lg:min-w-0',
                  'absolute h-full w-full lg:static lg:h-auto',
                  'border-0 border-marble-950 md:border-r',
                  'transition-[transform,min-width,max-width] duration-300 ease-in-out'
                )}
              >
                <ConversationListPanel agentId={agentId} />
              </Transition>
              <Transition
                as="main"
                show={isDesktop || !isMobileConvListPanelOpen}
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
                className={cn(
                  'flex min-w-0 flex-grow flex-col',
                  'transition-transform duration-500 ease-in-out'
                )}
              >
                {isLoading ? (
                  <div className="flex h-full flex-grow flex-col items-center justify-center">
                    <Spinner />
                  </div>
                ) : isError ? (
                  <ConversationError error={error} />
                ) : (
                  <Conversation
                    conversationId={conversationId}
                    agentId={agentId}
                    startOptionsEnabled
                  />
                )}
              </Transition>
            </div>
          </MainSection>
        </AgentsLayout>
      </ProtectedPage>
    );
  }

  return (
    <ProtectedPage>
      <Layout>
        <LeftSection>
          <ConversationListPanel />
        </LeftSection>
        <MainSection>
          <Conversation conversationId={conversation?.id} startOptionsEnabled />
        </MainSection>
      </Layout>
    </ProtectedPage>
  );
};

export default Page;
