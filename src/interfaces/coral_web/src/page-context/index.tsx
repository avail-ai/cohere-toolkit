import { QueryClient, useQueryClient } from '@tanstack/react-query';
import React, { PropsWithChildren, createContext, useState } from 'react';

import { CohereClient, useCohereClient } from '@/cohere-client';
import { useRouter } from 'next/router';

type Context = {
  queryClient: QueryClient;
  cohereClient: CohereClient;
}

const preloadPage = async (deps: Context) => {
  await Promise.allSettled([
    deps.queryClient.prefetchQuery({
      queryKey: ['conversations'],
      queryFn: async () => {
        return (await deps.cohereClient.listConversations({})) ?? [];
      },
    }),
    deps.queryClient.prefetchQuery({
      queryKey: ['tools'],
      queryFn: async () => await deps.cohereClient.listTools({}),
    }),
    deps.queryClient.prefetchQuery({
      queryKey: ['deployments'],
      queryFn: async () => await deps.cohereClient.listDeployments(),
    }),
  ]);

  return;
};

const loadConversation = async (deps: Context, conversationId: string) => {
  await Promise.allSettled([

    deps.queryClient.prefetchQuery({
      queryKey: ['conversation', conversationId],
      queryFn: async () => {
        const conversation = await deps.cohereClient.getConversation({
          conversationId,
        });
        // react-query useInfiniteQuery expected response shape
        return { conversation };
      },
    })]);

}

const PageContext = createContext<Context | undefined>(undefined);


const PageProvider: React.FC<PropsWithChildren> = ({ children }) => {
  const queryClient = useQueryClient();
  const cohereClient = useCohereClient();
  const [props, setProps] = useState<Context | undefined>(undefined);

  React.useEffect(() => {
    if (!queryClient || !cohereClient) return;

    preloadPage({ queryClient, cohereClient }).then(() => {
      setProps({ queryClient, cohereClient })
    })

  }, [queryClient, cohereClient])

  return (
    <PageContext.Provider value={props}>
      <>{children}</>
    </PageContext.Provider>
  );
};

export { PageProvider };
