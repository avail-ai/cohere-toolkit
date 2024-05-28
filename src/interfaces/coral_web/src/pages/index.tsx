import { NextPage } from 'next';
import { useContext, useEffect } from 'react';
import { useRouter } from 'next/router';

import { Document } from '@/cohere-client';
import Conversation from '@/components/Conversation';
import { ConversationError } from '@/components/ConversationError';
import ConversationListPanel from '@/components/ConversationList/ConversationListPanel';
import { Layout, LayoutSection } from '@/components/Layout';
import { Spinner } from '@/components/Shared';
import { BannerContext } from '@/context/BannerContext';
import { useConversation } from '@/hooks/conversation';
import { useListDeployments } from '@/hooks/deployments';
import { useExperimentalFeatures } from '@/hooks/experimentalFeatures';
import { useCitationsStore, useConversationStore, useParamsStore } from '@/stores';
import { createStartEndKey, mapHistoryToMessages } from '@/utils';
import { isEmpty } from 'lodash';


const ChatPage: NextPage = () => {
  const {
    setConversation,
    resetConversation,
  } = useConversationStore();
  const { addCitation, resetCitations } = useCitationsStore();
  const {
    params: { deployment },
    setParams,
  } = useParamsStore();
  const { data: availableDeployments } = useListDeployments();
  const { data: experimentalFeatures } = useExperimentalFeatures();
  const isLangchainModeOn = !!experimentalFeatures?.USE_EXPERIMENTAL_LANGCHAIN;
  const { setMessage } = useContext(BannerContext);
  const router = useRouter();

  const urlConversationId = Array.isArray(router.query.c)
    ? router.query.c[0]
    : (router.query.c as string);

  const {
    data: conversation,
    isLoading,
    isError,
    error,
  } = useConversation({ conversationId: urlConversationId });
  
  useEffect(() => {
    console.log("urlConversationId", urlConversationId)
    resetConversation();

    if (urlConversationId) {
      setConversation({ id: urlConversationId });
    }
  }, [urlConversationId, setConversation, resetCitations]);

  useEffect(() => {
    if (!conversation) return;

    const messages = mapHistoryToMessages(
      conversation?.messages?.sort((a, b) => a.position - b.position)
    );
    setConversation({ name: conversation.title, messages });
  }, [conversation?.id, setConversation]);

  useEffect(() => {
    console.log(deployment, availableDeployments)
    if (!deployment && availableDeployments && availableDeployments?.length > 0) {
      setParams({ deployment: availableDeployments[0].name });
    }
  }, [deployment, availableDeployments]);

  useEffect(() => {
    let documentsMap: { [documentId: string]: Document } = {};
    (conversation?.messages ?? []).forEach((message) => {
      documentsMap =
        message.documents?.reduce<{ [documentId: string]: Document }>(
          (idToDoc, doc) => ({ ...idToDoc, [doc.document_id ?? '']: doc }),
          {}
        ) ?? {};
      message.citations?.forEach((citation) => {
        const startEndKey = createStartEndKey(citation.start ?? 0, citation.end ?? 0);
        const documents = citation.document_ids?.map((id) => documentsMap[id]) ?? [];
        addCitation(message.generation_id ?? '', startEndKey, documents);
      });
    });
  }, [conversation]);

  useEffect(() => {
    if (!isLangchainModeOn) return;
    setMessage('You are using an experimental langchain multihop flow. There will be bugs.');
  }, [isLangchainModeOn]);

  return (
    <Layout>
      <LayoutSection.LeftDrawer>
        <ConversationListPanel />
      </LayoutSection.LeftDrawer>
      <LayoutSection.Main>
        {
          isLoading ? (
            <div className="flex h-full flex-grow flex-col items-center justify-center">
              <Spinner />
            </div>
          ) : isError ? (
            <ConversationError error={error} />
          ) :
            <Conversation conversationId={urlConversationId} startOptionsEnabled={isEmpty(urlConversationId)} />
        }
      </LayoutSection.Main>
    </Layout>
  );
};

export default ChatPage;
