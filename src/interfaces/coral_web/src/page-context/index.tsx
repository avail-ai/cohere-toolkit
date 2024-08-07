import { QueryClient, useQueryClient } from '@tanstack/react-query';
import React, { PropsWithChildren, createContext, useState } from 'react';

import { CohereClient, useCohereClient } from '@/cohere-client';

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
            queryFn: async () => await deps.cohereClient.listDeployments({ all: true }),
        }),
    ]);

    return;
};

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